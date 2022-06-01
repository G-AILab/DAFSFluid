import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


class Attention(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, 2)

    def forward(self, x):
        return self.linear1(x)


class AttentionCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, e_outputs, with_attention):
        super().__init__()
        self.obs_dim = obs_dim
        self.with_attention = with_attention
        # Layer E
        self.lineare = nn.Linear(obs_dim, e_outputs)
        # Layer 1
        self.linear1 = nn.Linear(obs_dim, hidden_sizes[0])
        # Layer 2
        # In the second layer the actions will be inserted also
        self.linear2 = nn.Linear(hidden_sizes[0] + act_dim, hidden_sizes[1])
        # Output layer (single value)
        self.V = nn.Linear(hidden_sizes[1], 1)
        # attention layers
        self.attention_layers = nn.ModuleList(
            [Attention(e_outputs) for _ in range(obs_dim)]
        )

    def forward(self, obs, act):
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

        # Layer 1
        x = self.linear1(state_attention)
        x = F.relu(x)
        # Layer 2
        x = torch.cat((x, act), 1)  # Insert the actions
        x = self.linear2(x)
        x = F.relu(x)
        # Output
        V = self.V(x)
        return V, Attention


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
        e_outputs=20,
        with_attention=False,
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        # actor net
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        # critic net
        self.q = AttentionCritic(
            obs_dim, act_dim, hidden_sizes, e_outputs, with_attention
        )

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
