import torch
import yaml
import env
import pandas as pd


def CUDA(tensor):
    return tensor.cuda() if torch.cuda.is_available() else tensor


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def load_config(config_path, profile=env.profile, set_probes_method=env.set_probes):
    """load configuration from yaml."""
    with open(config_path, "r") as f:
        cfgs = yaml.safe_load(f)

        dt = cfgs["solver_params"]["dt"]
        simulation_duration = cfgs["simulation_duration"]
        geometry_params = cfgs["geometry_params"]
        n_iter_length = cfgs["n_iter_length"]
        n_iter = int(n_iter_length / dt)
        size_his = int(simulation_duration / dt / 2)
        cfgs["flow_params"]["inflow_profile"] = profile
        cfgs["output_params"]["locations"] = set_probes_method(geometry_params)
        cfgs["n_iter"] = n_iter
        cfgs["size_his"] = size_his
        print(f"n_iter_make_ready: {n_iter}")
        print(f"history size: {size_his}")
        return cfgs


def reward_params(unconlog_path):
    C_Ds = [[] for _ in range(len(unconlog_path))]
    C_Ls = [[] for _ in range(len(unconlog_path))]
    for i, path in enumerate(unconlog_path):
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            if not line.startswith("Simu"):
                continue
            contents = line.split()
            C_Ds[i].append(abs(float(contents[11][:-1])))
            C_Ls[i].append(abs(float(contents[13][:-1])))
    C_Ddata = pd.DataFrame(C_Ds).T
    C_Ldata = pd.DataFrame(C_Ls).T
    C_Dmean = C_Ddata.mean()
    C_Lmean = C_Ldata.mean()
    reward_constent = C_Dmean.values
    reward_panalize = (C_Lmean / C_Dmean).values

    return reward_constent, reward_panalize


def str2bool(s: str):
    return True if s.lower() == "true" else False
