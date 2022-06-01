import argparse
import os
import pickle

import numpy as np
from dolfin import *

import env
from Env2DCylinder import RingBuffer
from flow_solver import FlowSolver
from generate_msh import generate_mesh
from msh_convert import convert
from probes import (
    PenetratedDragProbeANN,
    PenetratedLiftProbeANN,
    PressureProbeANN,
    RecirculationAreaProbe,
    VelocityProbeANN,
)
from utils import load_config


def write_history_parameters(
    history_params,
    len_jets,
    probe_type,
    len_probes,
    Qs,
    probes_values,
    drag,
    lift,
    recirc_area,
):
    for crrt_jet in range(len_jets):
        history_params["jet_{}".format(crrt_jet)].extend(Qs[crrt_jet])

    if probe_type == "pressure":
        for crrt_probe in range(len_probes):
            history_params["probe_{}".format(crrt_probe)].extend(
                probes_values[crrt_probe]
            )
    elif probe_type == "velocity":
        for crrt_probe in range(len_probes):
            history_params["probe_{}_u".format(crrt_probe)].extend(
                probes_values[2 * crrt_probe]
            )
            history_params["probe_{}_v".format(crrt_probe)].extend(
                history_params[2 * crrt_probe + 1]
            )

    history_params["drag"].extend(np.array(drag))
    history_params["lift"].extend(np.array(lift))
    history_params["recirc_area"].extend(np.array(recirc_area))


def make_mesh(
    geometry_params,
    flow_params,
    solver_params,
    output_params,
    dump=False,
    n_iter_make_ready=20000,
    size_history=2000,
    template="geometry_2d.template_geo",
    mesh_dir="mesh",
):
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
    mesh_name = "turek_2d"
    h5_path = ".".join([os.path.join(mesh_dir, mesh_name), "h5"])
    msh_path = ".".join([os.path.join(mesh_dir, mesh_name), "msh"])
    root_path = ".".join([os.path.join(mesh_dir, mesh_name), "geo"])
    geometry_params["output"] = root_path
    geometry_params["mesh"] = h5_path
    u_path = os.path.join(mesh_dir, "u_init.xdmf")
    p_path = os.path.join(mesh_dir, "p_init.xdmf")
    history_path = os.path.join(mesh_dir, "dict_history_parameters.pkl")

    history_params = {}
    len_jets = len(geometry_params["jet_positions"])
    probe_type = output_params["probe_type"]
    len_probes = len(output_params["locations"])

    solver_step = 0
    size_history = size_history

    # init history params
    for crrt_jet in range(len_jets):
        history_params["jet_{}".format(crrt_jet)] = RingBuffer(size_history)

    history_params["number_of_jets"] = len(geometry_params["jet_positions"])

    for crrt_probe in range(len_probes):
        if probe_type == "pressure":
            history_params["probe_{}".format(crrt_probe)] = RingBuffer(size_history)
        elif probe_type == "velocity":
            history_params["probe_{}_u".format(crrt_probe)] = RingBuffer(size_history)
            history_params["probe_{}_v".format(crrt_probe)] = RingBuffer(size_history)

    history_params["number_of_probes"] = len(output_params["locations"])
    history_params["drag"] = RingBuffer(size_history)
    history_params["lift"] = RingBuffer(size_history)
    history_params["recirc_area"] = RingBuffer(size_history)

    # generate mesh
    generate_mesh(geometry_params, template=template)
    assert os.path.exists(msh_path)

    convert(msh_path, h5_path)
    assert os.path.exists(h5_path)

    # create the flow simulation object
    flow_solver = FlowSolver(flow_params, geometry_params, solver_params)
    # Setup probes
    if output_params["probe_type"] == "pressure":
        ann_probes = PressureProbeANN(flow_solver, output_params["locations"])
    elif output_params["probe_type"] == "velocity":
        ann_probes = VelocityProbeANN(flow_solver, output_params["locations"])
    else:
        raise RuntimeError("unknown probe type")

    # Setup drag measurement
    drag_probe = PenetratedDragProbeANN(flow_solver)
    lift_probe = PenetratedLiftProbeANN(flow_solver)

    # No flux from jets for starting
    Qs = np.zeros(len(geometry_params["jet_positions"]))
    action = np.zeros(len(geometry_params["jet_positions"]))

    u_, p_ = flow_solver.evolve(Qs)
    dump_path = os.path.join("results", "area_out.pvd") if dump else ""
    area_probe = RecirculationAreaProbe(u_, 0, store_path=dump_path)

    for _ in range(n_iter_make_ready):
        u_, p_ = flow_solver.evolve(Qs)

        probes_values = ann_probes.sample(u_, p_).flatten()
        drag = drag_probe.sample(u_, p_)
        lift = lift_probe.sample(u_, p_)
        recirc_area = area_probe.sample(u_, p_)

        write_history_parameters(
            history_params,
            len_jets,
            probe_type,
            len_probes,
            Qs,
            probes_values,
            drag,
            lift,
            recirc_area,
        )
        # visual_inspection()
        # output_data()

        solver_step += 1

    encoding = XDMFFile.Encoding.HDF5
    mesh = convert(msh_path, h5_path)
    comm = mesh.mpi_comm()

    # save field data
    XDMFFile(comm, u_path).write_checkpoint(u_, "u0", 0, encoding)
    XDMFFile(comm, p_path).write_checkpoint(p_, "p0", 0, encoding)

    # save buffer dict
    with open(history_path, "wb") as f:
        pickle.dump(history_params, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="re100.yaml")
    args = parser.parse_args()

    cfgs = load_config(args.config_path)

    make_mesh(
        cfgs["geometry_params"],
        cfgs["flow_params"],
        cfgs["solver_params"],
        cfgs["output_params"],
        dump=cfgs["inspection_params"]["dump"],
        n_iter_make_ready=cfgs["n_iter"],
        size_history=cfgs["size_his"],
        mesh_dir=cfgs["mesh_dir"],
    )
