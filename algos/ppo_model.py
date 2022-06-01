import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
from torch.nn.modules import activation


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1])]
        if j > 0 and j < len(sizes) - 2:
            d = nn.Dropout(p=0.5)
            layers.append(d)
        layers.append(act())
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        # init layer
        for layer in self.mu_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        # Last axis sum needed for Torch Normal distribution
        return pi.log_prob(act).sum(axis=-1)


class MLPBetaActor(Actor):
    def __init__(self, obs_dim, act_dim, act_limit, hidden_size, activation=F.softplus):
        super().__init__()
        self.act_limit = act_limit
        self.alpha = mlp([obs_dim] + [hidden_size] + [act_dim], nn.Identity, activation)
        self.beta = mlp([obs_dim] + [hidden_size] + [act_dim], nn.Identity, activation)
        # init layer
        for layer in self.alpha:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
        for layer in self.beta:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def _distribution(self, obs):
        action_a = self.alpha(obs) + 1
        action_b = self.beta(obs) + 1
        return Beta(action_a, action_b, True)

    def _log_prob_from_distribution(self, pi, act):
        act = (act + self.act_limit) / (2 * self.act_limit)
        return pi.log_prob(act).sum(-1)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
        # init layer
        for layer in self.v_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, obs):
        # Critical to ensure v has right shape.
        return torch.squeeze(self.v_net(obs), -1)


class Attention(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, 2)
        nn.init.kaiming_normal_(self.linear1.weight)

    def forward(self, x):
        return self.linear1(x)


class AttentionCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation, e_outputs, with_attention):
        super().__init__()
        self.obs_dim = obs_dim
        self.with_attention = with_attention
        # Layer E
        self.lineare = nn.Linear(obs_dim, e_outputs)
        # attention layers
        self.attention_layers = nn.ModuleList(
            [Attention(e_outputs) for _ in range(obs_dim)]
        )
        # mlp layers
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
        # init layer
        for layer in self.v_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.lineare.weight)

    def forward(self, obs):
        e = obs
        state_attention = obs
        Attention = None
        if self.with_attention:
            # Layer E
            e = self.lineare(e)
            e = F.tanh(e)
            attention_out_list = []
            for i in range(self.obs_dim):
                attention_FC = self.attention_layers[i](e)
                attention_out = F.softmax(attention_FC, dim=1)
                attention_out_list.append(attention_out[:, 1])
            Attention = torch.stack(attention_out_list).T
            state_attention = torch.mul(obs, Attention)
        # mlp layers
        v = torch.squeeze(self.v_net(state_attention), -1)
        return v, Attention


class MLPActorCritic(nn.Module):
    def __init__(
        self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(
                obs_dim, action_space.shape[0], hidden_sizes, activation
            )
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation
            )

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            # deterministic
            a = pi.mean
            # a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            a = a.numpy()
            a = 0.01 * a
            a = np.clip(a, -0.01, 0.01)
        return a, v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class AttentionActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(64, 64),
        activation=nn.Tanh,
        e_outputs=20,
        with_attention=False,
        deterministic=True,
    ):
        """
        e_outputs: num of attention layer weights
        with_attention: with or without attention layer
        deterministic: actor selection method,
                       in the training process we use stochastic policy,
                       in the testing process we use deterministic policy.
        """
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]
        self.deterministic = deterministic

        # policy builder depends on action space
        if isinstance(action_space, Box):
            # self.pi = MLPGaussianActor(
            #     obs_dim, action_space.shape[0], hidden_sizes, activation
            # )
            # beta policy for continuous limited action space
            self.pi = MLPBetaActor(
                obs_dim, act_dim, self.act_limit, hidden_sizes[0], nn.Softplus
            )
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation
            )

        # build value function
        self.v = AttentionCritic(
            obs_dim,
            hidden_sizes,
            activation,
            e_outputs=e_outputs,
            with_attention=with_attention,
        )

    # def step(self, obs):
    #     with torch.no_grad():
    #         pi = self.pi._distribution(obs)
    #         # deterministic
    #         a = pi.mean
    #         # a = pi.sample()
    #         logp_a = self.pi._log_prob_from_distribution(pi, a)
    #         v, Attention = self.v(obs)
    #         a = a.numpy()
    #         a = 0.01 * a
    #         a = np.clip(a, -0.01, 0.01)
    #     return a, v.numpy(), logp_a.numpy(), Attention

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.mean if self.deterministic else pi.sample()
            a = -self.act_limit + (self.act_limit + self.act_limit) * a
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            a = a.numpy()
            a = np.clip(a, -self.act_limit, self.act_limit)
            v, Attention = self.v(obs)
        return a, v.numpy(), logp_a.numpy(), Attention

    def act(self, obs):
        return self.step(obs)[0]
