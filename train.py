from utils import load_config, str2bool
from algos.ppo_model import MLPActorCritic, AttentionActorCritic
from algos.ppo import ppo
import gym
from algos.mpi_tools import mpi_fork
from env import resume_env
import torch.nn as nn
import numpy as np
import time


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--with_attention", type=str2bool, default="false")
    parser.add_argument("--config_path", type=str, default="re40.yaml")
    parser.add_argument("--use_selection", type=str2bool, default="false")
    parser.add_argument("--att_path", type=str, default="att.csv")
    parser.add_argument("--num_selection", type=int, default=5)
    parser.add_argument("--model_dir", type=str, default="re100/models")
    args = parser.parse_args()

    # run parallel code with mpi
    mpi_fork(args.cpu)

    # env generation method
    environment = resume_env

    # random probes index
    random_probes_indexs = [
        [144, 105, 86, 10, 126],
        [136, 69, 37, 148, 26],
        [25, 138, 147, 76, 136],
        [3, 67, 147, 41, 89],
        [112, 145, 33, 130, 122],
    ]

    # rabault paper's 5 selection
    paper_5_probes_index = np.array([1, 3, 5, 88, 106])
    # rabault paper's 11 selection
    paper_11_probes_index = np.array([0, 3, 6, 21, 24, 27, 42, 45, 48, 88, 106])
    # randomly selection 50 probes to test its time complexity
    time_50_probes_index = np.array([x for x in range(50)])
    # random noise probe selection
    noise_probe_index = np.array(
        [103, 85, 94, 10, 66, 12, 7, 5, 121, 114, 35, 123, 122, 49, 56]
    )

    # FIR
    fir_probes_selection = [82, 88, 104, 106, 112]
    # fir_probes_selection = [21, 66, 110, 113, 120]
    # ccm
    ccm_probes_selection = [79, 3, 80, 52, 115]
    # l2x
    # l2x_probes_seleoction = [88, 48, 39, 12, 55]
    l2x_probes_selection = [39, 131, 68, 107, 9]
    # san global
    san_g_probes_selection = [96, 140, 120, 37, 124]
    # san local
    san_l_probes_selection = [105, 96, 100, 67, 149]
    # xgb
    xgb_probes_selection = [79, 82, 87, 112, 34]
    # lgb
    lgb_probes_selection = [87, 79, 107, 8, 12]
    # rf
    rf_probes_selection = [125, 134, 127, 132, 141]
    # fisherscore
    fisher_probes_selection = [0, 98, 92, 93, 94]

    index_selection = np.array(rf_probes_selection)

    start_time = time.time()

    cfgs = load_config(args.config_path)
    ac_net_params = cfgs["ac_net_params"]
    training_params = cfgs["training_params"]
    training_params["model_dir"] = args.model_dir

    ppo(
        env_fn=environment,
        # env args
        # geometry_params=geometry_params,
        env_kwargs=dict(
            cfgs=cfgs,
            use_selection=args.use_selection,
            attention_path=args.att_path,
            num_selection=args.num_selection,
            index_selection=index_selection,
            noise_probe_index=None,
        ),
        actor_critic=AttentionActorCritic,
        # ac network args
        ac_kwargs=dict(
            hidden_sizes=[ac_net_params["hid"]] * ac_net_params["l"],
            e_outputs=ac_net_params["e_outputs"],
            with_attention=args.with_attention,
            activation=nn.LeakyReLU,
            deterministic=False,
        ),
        **training_params
    )

    end_time = time.time()
    print(end_time - start_time)
