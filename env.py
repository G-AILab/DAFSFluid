"""Resume and use the environment.
"""

from pickle import NONE
import sys
import os
import shutil

cwd = os.getcwd()
sys.path.append(cwd + "/../")

from Env2DCylinder import Env2DCylinder
import numpy as np
from dolfin import Expression

# from printind.printind_function import printi, printiv
import math

import os

import pandas as pd

cwd = os.getcwd()

# nb_actuations = 80  # Nombre d'actuations du reseau de neurones par episode


def set_probes(geometry_params):
    """Set 151 probe param."""
    list_position_probes = []

    positions_probes_for_grid_x = [0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    positions_probes_for_grid_y = [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]
    # positions_probes_for_grid_y = [0.15]

    # 200 probes
    # positions_probes_for_grid_x = [0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
    # positions_probes_for_grid_y = [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]

    for crrt_x in positions_probes_for_grid_x:
        for crrt_y in positions_probes_for_grid_y:
            list_position_probes.append(np.array([crrt_x, crrt_y]))

    positions_probes_for_grid_x = [-0.025, 0.0, 0.025, 0.05]
    positions_probes_for_grid_y = [-0.15, -0.1, 0.1, 0.15]

    for crrt_x in positions_probes_for_grid_x:
        for crrt_y in positions_probes_for_grid_y:
            list_position_probes.append(np.array([crrt_x, crrt_y]))

    list_radius_around = [
        geometry_params["jet_radius"] + 0.02,
        geometry_params["jet_radius"] + 0.05,
    ]
    list_angles_around = np.arange(0, 360, 10)

    for crrt_radius in list_radius_around:
        for crrt_angle in list_angles_around:
            angle_rad = np.pi * crrt_angle / 180.0
            list_position_probes.append(
                np.array(
                    [
                        crrt_radius * math.cos(angle_rad),
                        crrt_radius * math.sin(angle_rad),
                    ]
                )
            )
    return list_position_probes


def profile(mesh, degree):
    bot = mesh.coordinates().min(axis=0)[1]
    top = mesh.coordinates().max(axis=0)[1]
    print(bot, top)
    H = top - bot

    Um = 1.5

    return Expression(
        ("-4*Um*(x[1]-bot)*(x[1]-top)/H/H", "0"),
        bot=bot,
        top=top,
        H=H,
        Um=Um,
        degree=degree,
    )


def resume_env(
    cfgs,
    use_selection=False,
    attention_path=None,
    num_selection=20,
    index_selection=None,
    noise_probe_index=None,
):
    """
    Start CFD env.
    nb_actuations: Nombre d'actuations du reseau de neurones par episode
    simulation_duration: duree en secondes de la simulation
    dt = discrete time

    use_selection: if or not use selected probes
    attention_path: attention.csv file path, contains 151 probes weights
    num_selection: how many probes you want to select
    index_selection: if not None, use it as the selected index, and don't use attention result
    """
    # ---------------------------------------------------------------------------------
    # the configuration version number 1

    simulation_duration = cfgs[
        "simulation_duration"
    ]  # duree en secondes de la simulation
    nb_actuations = cfgs["nb_actuations"]
    dt = cfgs["solver_params"]["dt"]
    mesh_dir = cfgs["mesh_dir"]

    root = os.path.join(mesh_dir, "turek_2d")
    if not os.path.exists(mesh_dir):
        os.mkdir(mesh_dir)

    geometry_params = cfgs["geometry_params"]
    flow_params = cfgs["flow_params"]
    solver_params = cfgs["solver_params"]
    output_params = cfgs["output_params"]
    optimization_params = cfgs["optimization_params"]
    inspection_params = cfgs["inspection_params"]
    reward_function = cfgs["reward_function"]
    drag_constant = cfgs["drag_constant"]
    lift_penalize = cfgs["lift_penalize"]

    verbose = 0

    number_steps_execution = int((simulation_duration / dt) / nb_actuations)

    # we don't make mesh here
    n_iter = None

    # Processing the name of the simulation
    simu_name = "Simu"

    if (geometry_params["jet_positions"][0] - 90) != 0:
        next_param = "A" + str(geometry_params["jet_positions"][0] - 90)
        simu_name = "_".join([simu_name, next_param])
    if geometry_params["cylinder_size"] != 0.01:
        next_param = "M" + str(geometry_params["cylinder_size"])[2:]
        simu_name = "_".join([simu_name, next_param])
    if optimization_params["max_value_jet_MFR"] != 0.01:
        next_param = "maxF" + str(optimization_params["max_value_jet_MFR"])[2:]
        simu_name = "_".join([simu_name, next_param])
    if nb_actuations != 80:
        next_param = "NbAct" + str(nb_actuations)
        simu_name = "_".join([simu_name, next_param])
    next_param = "drag"
    if reward_function == "recirculation_area":
        next_param = "area"
    if reward_function == "max_recirculation_area":
        next_param = "max_area"
    elif reward_function == "drag":
        next_param = "last_drag"
    elif reward_function == "max_plain_drag":
        next_param = "max_plain_drag"
    elif reward_function == "drag_plain_lift":
        next_param = "lift"
    elif reward_function == "drag_avg_abs_lift":
        next_param = "avgAbsLift"
    simu_name = "_".join([simu_name, next_param])

    if use_selection and index_selection is None:
        attention_weight_result = pd.read_csv(attention_path, header=None)
        attention_weight_result = attention_weight_result[-1:]
        attention_weight_result = np.array(attention_weight_result)[0]
        index_selection = attention_weight_result.argsort()[::-1][0:num_selection]

    env_2d_cylinder = Env2DCylinder(
        mesh_dir=mesh_dir,
        path_root=root,
        geometry_params=geometry_params,
        flow_params=flow_params,
        solver_params=solver_params,
        output_params=output_params,
        optimization_params=optimization_params,
        inspection_params=inspection_params,
        n_iter_make_ready=n_iter,  # On recalcule si besoin
        verbose=verbose,
        reward_function=reward_function,
        drag_constant=drag_constant,
        lift_penalize=lift_penalize,
        number_steps_execution=number_steps_execution,
        simu_name=simu_name,
        use_selection=use_selection,
        obs_dim=num_selection,
        index_selection=index_selection,
        noise_probe_index=noise_probe_index,
    )

    return env_2d_cylinder


if __name__ == "__main__":
    resume_env()
