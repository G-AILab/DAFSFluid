import argparse
import os

import numpy as np

import env
from algos.ppo import test
from env import resume_env
from utils import CUDA, load_config, str2bool


def test_cylinder(env, episodes_n):
    """
    test in the case without control
    """
    # 11, 15, 28, 113, 117
    # probes = [11, 15, 28, 113, 117]
    # states = []
    for _ in range(episodes_n):
        # a = env.action_space.sample()
        a = np.array([0, 0])
        s_, _, _ = env.step(a)
    #     states.append(s_)
    # a = np.stack(states, axis=1)
    # b = a.std(1)
    # for probe in probes:
    #     print(f"Var: {probe}: {b[probe]}")
    # print(f"Min Var: {b.min()}")
    # print(f"Mean Var: {b.mean()}")
    # print(f"Max Var: {b.max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uncontrolled", type=str2bool, default="true")
    parser.add_argument("--config_path", type=str, default="re80.yaml")
    parser.add_argument("--att_path", type=str, default="attention.csv")
    parser.add_argument("--model_path", type=str, default="att.model")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--use_selection", type=str2bool, default="false")
    parser.add_argument("--num_selection", type=int, default=5)
    args = parser.parse_args()

    random_probes_indexs = [
        [144, 105, 86, 10, 126],
        [136, 69, 37, 148, 26],
        [25, 138, 147, 76, 136],
        [3, 67, 147, 41, 89],
        [112, 145, 33, 130, 122],
    ]

    noise_probe_index = np.array(
        [103, 85, 94, 10, 66, 12, 7, 5, 121, 114, 35, 123, 122, 49, 56]
    )

    paper_5_probes_index = np.array([1, 3, 5, 88, 106])
    paper_11_probes_index = np.array([0, 3, 6, 21, 24, 27, 42, 45, 48, 88, 106])

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

    cfgs = load_config(args.config_path)

    environment = resume_env(
        cfgs,
        use_selection=args.use_selection,
        attention_path=args.att_path,
        num_selection=args.num_selection,
        index_selection=index_selection,
        noise_probe_index=None,
    )

    if args.uncontrolled:
        test_cylinder(environment, 100)
    else:
        test(
            environment,
            args.epochs,
            model_path=args.model_path,
            att_path=args.att_path,
        )
