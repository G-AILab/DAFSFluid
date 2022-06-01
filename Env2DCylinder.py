# from printind.printind_decorators import printi_all_method_calls as printidc
# from printind.printind_function import printi, printiv
# from tensorforce.environments import Environment
# import tensorforce
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from threading import Thread
# from tensorforce import TensorforceError

# a bit hacky, but meeehh... FIXME!!
import sys
import os

cwd = os.getcwd()
sys.path.append(cwd + "/../Simulation/")

from dolfin import Expression, File, plot
from probes import (
    PenetratedDragProbeANN,
    PenetratedLiftProbeANN,
    PressureProbeANN,
    VelocityProbeANN,
    RecirculationAreaProbe,
)
from generate_msh import generate_mesh
from flow_solver import FlowSolver
from msh_convert import convert
from dolfin import *

import numpy as np
import os

import pickle

import time
import math
import csv

import shutil

from gym.spaces import Box

# TODO: check that right types etc from tensorfoce examples
# typically:

# from tensorforce.contrib.openai_gym import OpenAIGym
# environment = OpenAIGym('MountainCarContinuous-v0', visualize=False)

# printiv(environment.states)
# environment.states = {'shape': (2,), 'type': 'float'}
# printiv(environment.actions)
# environment.actions = {'max_value': 1.0, 'shape': (1,), 'min_value': -1.0, 'type': 'float'}


def constant_profile(mesh, degree):
    """
    Time independent inflow profile.
    """
    bot = mesh.coordinates().min(axis=0)[1]
    top = mesh.coordinates().max(axis=0)[1]

    H = top - bot

    Um = 1.5

    return Expression(
        ("-4*Um*(x[1]-bot)*(x[1]-top)/H/H", "0"),
        bot=bot,
        top=top,
        H=H,
        Um=Um,
        degree=degree,
        time=0,
    )


class RingBuffer:
    "A 1D ring buffer using numpy arrays"

    def __init__(self, length):
        self.data = np.zeros(length, dtype="f")
        self.index = 0

    def extend(self, x):
        "adds array x to ring buffer"
        x_index = (self.index + np.arange(x.size)) % self.data.size
        self.data[x_index] = x
        self.index = x_index[-1] + 1

    def get(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]


# @printidc()
class Env2DCylinder():
    """Environment for 2D flow simulation around a cylinder."""

    def __init__(
        self,
        mesh_dir,
        path_root,
        geometry_params,
        flow_params,
        solver_params,
        output_params,
        optimization_params,
        inspection_params,
        n_iter_make_ready=None,
        verbose=0,
        size_history=2000,
        reward_function="plain_drag",
        drag_constant=0.159,
        lift_penalize=0.2,
        size_time_state=50,
        number_steps_execution=1,
        simu_name="Simu",
        use_selection=False,
        obs_dim=None,
        index_selection=None,
        noise_probe_index=None,
    ):
        """ """

        # TODO: should actually save the dicts in to double check when loading that using compatible simulations together

        # printi("--- call init ---")

        self.observation = None
        self.thread = None

        self.mesh_dir = mesh_dir
        self.path_root = path_root
        self.flow_params = flow_params
        self.geometry_params = geometry_params
        self.solver_params = solver_params
        self.output_params = output_params
        self.optimization_params = optimization_params
        self.inspection_params = inspection_params
        self.verbose = verbose
        self.n_iter_make_ready = n_iter_make_ready
        self.size_history = size_history
        self.reward_function = reward_function
        self.drag_constant = drag_constant
        self.lift_penalize = lift_penalize
        self.size_time_state = size_time_state
        self.number_steps_execution = number_steps_execution

        self.simu_name = simu_name

        self.use_selection = use_selection
        self.index_selection = index_selection
        self.noise_probe_index = noise_probe_index

        obs_dim = len(self.output_params["locations"]) if not use_selection else obs_dim
        act_dim = len(self.geometry_params["jet_positions"])
        obs_low = np.array([-1] * obs_dim)
        obs_high = np.array([2] * obs_dim)
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)
        act_low = np.array([self.optimization_params["min_value_jet_MFR"]] * act_dim)
        act_high = np.array([self.optimization_params["max_value_jet_MFR"]] * act_dim)
        self.action_space = Box(low=act_low, high=act_high, dtype=np.float32)

        # Relatif a l'ecriture des .csv
        name = "output.csv"
        last_row = None
        if os.path.exists("saved_models/" + name):
            with open("saved_models/" + name, "r") as f:
                for row in reversed(
                    list(csv.reader(f, delimiter=";", lineterminator="\n"))
                ):
                    last_row = row
                    break
        if not last_row is None:
            self.episode_number = int(last_row[0])
            self.last_episode_number = int(last_row[0])
        else:
            self.last_episode_number = 0
            self.episode_number = 0
        self.episode_drags = np.array([])
        self.episode_areas = np.array([])
        self.episode_lifts = np.array([])

        self.initialized_visualization = False

        self.start_class()

        # printi("--- done init ---")

    def start_class(self):
        self.solver_step = 0
        self.accumulated_drag = 0
        self.accumulated_lift = 0

        self.initialized_output = False

        self.resetted_number_probes = False

        self.area_probe = None

        self.history_parameters = {}

        for crrt_jet in range(len(self.geometry_params["jet_positions"])):
            self.history_parameters["jet_{}".format(crrt_jet)] = RingBuffer(
                self.size_history
            )

        self.history_parameters["number_of_jets"] = len(
            self.geometry_params["jet_positions"]
        )

        for crrt_probe in range(len(self.output_params["locations"])):
            if self.output_params["probe_type"] == "pressure":
                self.history_parameters["probe_{}".format(crrt_probe)] = RingBuffer(
                    self.size_history
                )
            elif self.output_params["probe_type"] == "velocity":
                self.history_parameters["probe_{}_u".format(crrt_probe)] = RingBuffer(
                    self.size_history
                )
                self.history_parameters["probe_{}_v".format(crrt_probe)] = RingBuffer(
                    self.size_history
                )

        self.history_parameters["number_of_probes"] = len(
            self.output_params["locations"]
        )

        self.history_parameters["drag"] = RingBuffer(self.size_history)
        self.history_parameters["lift"] = RingBuffer(self.size_history)
        self.history_parameters["recirc_area"] = RingBuffer(self.size_history)

        # ------------------------------------------------------------------------
        # remesh if necessary
        mesh_name = "turek_2d"
        h5_file = ".".join([os.path.join(self.mesh_dir, mesh_name), "h5"])
        msh_file = ".".join([os.path.join(self.mesh_dir, mesh_name), "msh"])
        self.geometry_params["mesh"] = h5_file

        root_path = ".".join([os.path.join(self.mesh_dir, mesh_name), "geo"])
        u_path = os.path.join(self.mesh_dir, "u_init.xdmf")
        p_path = os.path.join(self.mesh_dir, "p_init.xdmf")
        history_path = os.path.join(self.mesh_dir, "dict_history_parameters.pkl")

        # Regenerate mesh?
        if self.geometry_params["remesh"]:

            if self.verbose > 0:
                print("Remesh")
                # printi("generate_mesh start...")

            generate_mesh(
                self.geometry_params, template=self.geometry_params["template"]
            )

            if self.verbose > 0:
                print("generate_mesh done!")
            print(msh_file)
            assert os.path.exists(msh_file)

            convert(msh_file, h5_file)
            assert os.path.exists(h5_file)

        # ------------------------------------------------------------------------
        # if necessary, load initialization fields
        if self.n_iter_make_ready is None:
            if self.verbose > 0:
                print("Load initial flow")

            self.flow_params["u_init"] = u_path
            self.flow_params["p_init"] = p_path

            if self.verbose > 0:
                print("Load buffer history")

            with open(history_path, "rb") as f:
                self.history_parameters = pickle.load(f)

            if not "number_of_probes" in self.history_parameters:
                self.history_parameters["number_of_probes"] = 0

            if not "number_of_jets" in self.history_parameters:
                self.history_parameters["number_of_jets"] = len(
                    self.geometry_params["jet_positions"]
                )
                # printi("Warning!! The number of jets was not set in the loaded hdf5 file")

            if not "lift" in self.history_parameters:
                self.history_parameters["lift"] = RingBuffer(self.size_history)
                # printi("Warning!! No value for the lift founded")

            if not "recirc_area" in self.history_parameters:
                self.history_parameters["recirc_area"] = RingBuffer(self.size_history)
                # printi("Warning!! No value for the recirculation area founded")

            # if not the same number of probes, reset
            if not self.history_parameters["number_of_probes"] == len(
                self.output_params["locations"]
            ):
                for crrt_probe in range(len(self.output_params["locations"])):
                    if self.output_params["probe_type"] == "pressure":
                        self.history_parameters[
                            "probe_{}".format(crrt_probe)
                        ] = RingBuffer(self.size_history)
                    elif self.output_params["probe_type"] == "velocity":
                        self.history_parameters[
                            "probe_{}_u".format(crrt_probe)
                        ] = RingBuffer(self.size_history)
                        self.history_parameters[
                            "probe_{}_v".format(crrt_probe)
                        ] = RingBuffer(self.size_history)

                self.history_parameters["number_of_probes"] = len(
                    self.output_params["locations"]
                )

                # printi("Warning!! Number of probes was changed! Probes buffer content reseted")

                self.resetted_number_probes = True

        # ------------------------------------------------------------------------
        # create the flow simulation object
        self.flow = FlowSolver(
            self.flow_params, self.geometry_params, self.solver_params
        )

        # ------------------------------------------------------------------------
        # Setup probes
        if self.output_params["probe_type"] == "pressure":
            self.ann_probes = PressureProbeANN(
                self.flow, self.output_params["locations"]
            )

        elif self.output_params["probe_type"] == "velocity":
            self.ann_probes = VelocityProbeANN(
                self.flow, self.output_params["locations"]
            )
        else:
            raise RuntimeError("unknown probe type")

        # Setup drag measurement
        self.drag_probe = PenetratedDragProbeANN(self.flow)
        self.lift_probe = PenetratedLiftProbeANN(self.flow)

        # ------------------------------------------------------------------------
        # No flux from jets for starting
        self.Qs = np.zeros(len(self.geometry_params["jet_positions"]))
        self.action = np.zeros(len(self.geometry_params["jet_positions"]))

        # ------------------------------------------------------------------------
        # prepare the arrays for plotting positions
        self.compute_positions_for_plotting()

        # ------------------------------------------------------------------------
        # if necessary, make converge
        if self.n_iter_make_ready is not None:
            self.u_, self.p_ = self.flow.evolve(self.Qs)
            path = ""
            if "dump" in self.inspection_params:
                path = "results/area_out.pvd"
            self.area_probe = RecirculationAreaProbe(self.u_, 0, store_path=path)
            if self.verbose > 0:
                print("Compute initial flow")
                # printiv(self.n_iter_make_ready)

            for _ in range(self.n_iter_make_ready):
                self.u_, self.p_ = self.flow.evolve(self.Qs)

                self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()
                self.drag = self.drag_probe.sample(self.u_, self.p_)
                self.lift = self.lift_probe.sample(self.u_, self.p_)
                self.recirc_area = self.area_probe.sample(self.u_, self.p_)

                self.write_history_parameters()
                self.visual_inspection()
                self.output_data()

                self.solver_step += 1

        if self.n_iter_make_ready is not None:
            encoding = XDMFFile.Encoding.HDF5
            mesh = convert(msh_file, h5_file)
            comm = mesh.mpi_comm()

            # save field data
            XDMFFile(comm, u_path).write_checkpoint(
                self.u_, "u0", 0, encoding
            )
            XDMFFile(comm, p_path).write_checkpoint(
                self.p_, "p0", 0, encoding
            )

            # save buffer dict
            with open(history_path, "wb") as f:
                pickle.dump(self.history_parameters, f, pickle.HIGHEST_PROTOCOL)

        # ----------------------------------------------------------------------
        # if reading from disk, show to check everything ok
        if self.n_iter_make_ready is None:
            # Let's start in a random position of the vortex shading
            if self.optimization_params["random_start"]:
                rd_advancement = np.random.randint(650)
                for j in range(rd_advancement):
                    self.flow.evolve(self.Qs)
                print(
                    "Simulated {} iterations before starting the control".format(
                        rd_advancement
                    )
                )

            self.u_, self.p_ = self.flow.evolve(self.Qs)
            path = ""
            if "dump" in self.inspection_params:
                path = "results/area_out.pvd"
            self.area_probe = RecirculationAreaProbe(self.u_, 0, store_path=path)

            self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()
            self.drag = self.drag_probe.sample(self.u_, self.p_)
            self.lift = self.lift_probe.sample(self.u_, self.p_)
            self.recirc_area = self.area_probe.sample(self.u_, self.p_)

            self.write_history_parameters()
            # self.visual_inspection()
            # self.output_data()

            # self.solver_step += 1

            # time.sleep(10)

        # ----------------------------------------------------------------------
        # if necessary, fill the probes buffer
        if self.resetted_number_probes:
            # printi("Need to fill again the buffer; modified number of probes")

            for _ in range(self.size_history):
                self.execute()

        # ----------------------------------------------------------------------
        # ready now

        # Initialisation du prob de recirculation area
        # path=''
        # if "dump" in self.inspection_params:
        #    path = 'results/area_out.pvd'
        # self.area_probe = RecirculationAreaProbe(self.u_, 0, store_path=path)

        self.ready_to_use = True

    def write_history_parameters(self):
        for crrt_jet in range(len(self.geometry_params["jet_positions"])):
            self.history_parameters["jet_{}".format(crrt_jet)].extend(self.Qs[crrt_jet])

        if self.output_params["probe_type"] == "pressure":
            for crrt_probe in range(len(self.output_params["locations"])):
                self.history_parameters["probe_{}".format(crrt_probe)].extend(
                    self.probes_values[crrt_probe]
                )
        elif self.output_params["probe_type"] == "velocity":
            for crrt_probe in range(len(self.output_params["locations"])):
                self.history_parameters["probe_{}_u".format(crrt_probe)].extend(
                    self.probes_values[2 * crrt_probe]
                )
                self.history_parameters["probe_{}_v".format(crrt_probe)].extend(
                    self.probes_values[2 * crrt_probe + 1]
                )

        self.history_parameters["drag"].extend(np.array(self.drag))
        self.history_parameters["lift"].extend(np.array(self.lift))
        self.history_parameters["recirc_area"].extend(np.array(self.recirc_area))

    def compute_positions_for_plotting(self):
        # where the pressure probes are
        self.list_positions_probes_x = []
        self.list_positions_probes_y = []

        # total_number_of_probes = len(self.output_params['locations'])

        # printiv(total_number_of_probes)

        # get the positions
        for crrt_probe in self.output_params["locations"]:
            if self.verbose > 2:
                print(crrt_probe)

            self.list_positions_probes_x.append(crrt_probe[0])
            self.list_positions_probes_y.append(crrt_probe[1])

        # where the jets are
        radius_cylinder = (
            self.geometry_params["cylinder_size"]
            / 2.0
            / self.geometry_params["clscale"]
        )
        self.list_positions_jets_x = []
        self.list_positions_jets_y = []

        # compute the positions
        for crrt_jet_angle in self.geometry_params["jet_positions"]:
            crrt_jet_angle_rad = math.pi / 180.0 * crrt_jet_angle
            crrt_x = radius_cylinder * math.cos(crrt_jet_angle_rad)
            crrt_y = radius_cylinder * math.sin(crrt_jet_angle_rad)
            self.list_positions_jets_x.append(crrt_x)
            self.list_positions_jets_y.append(1.1 * crrt_y)

    def show_flow(self):
        plt.figure()
        plot(self.u_)
        plt.scatter(
            self.list_positions_probes_x,
            self.list_positions_probes_y,
            c="k",
            marker="o",
        )
        plt.scatter(
            self.list_positions_jets_x, self.list_positions_jets_y, c="r", marker="o"
        )
        plt.xlim(
            [
                -self.geometry_params["front_distance"],
                self.geometry_params["length"] - self.geometry_params["front_distance"],
            ]
        )
        plt.ylim(
            [
                -self.geometry_params["bottom_distance"],
                self.geometry_params["width"] - self.geometry_params["bottom_distance"],
            ]
        )
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.show()

        plt.figure()
        p = plot(self.p_)
        cb = plt.colorbar(p, fraction=0.1, shrink=0.3)
        plt.scatter(
            self.list_positions_probes_x,
            self.list_positions_probes_y,
            c="k",
            marker="o",
        )
        plt.scatter(
            self.list_positions_jets_x, self.list_positions_jets_y, c="r", marker="o"
        )
        plt.xlim(
            [
                -self.geometry_params["front_distance"],
                self.geometry_params["length"] - self.geometry_params["front_distance"],
            ]
        )
        plt.ylim(
            [
                -self.geometry_params["bottom_distance"],
                self.geometry_params["width"] - self.geometry_params["bottom_distance"],
            ]
        )
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.tight_layout()
        cb.set_label("P")
        plt.show()

    def show_control(self):
        plt.figure()

        linestyles = ["-", "--", ":", "-."]

        for crrt_jet in range(len(self.geometry_params["jet_positions"])):
            crrt_jet_data = self.history_parameters["jet_{}".format(crrt_jet)].get()
            plt.plot(
                crrt_jet_data,
                label="jet {}".format(crrt_jet),
                linestyle=linestyles[crrt_jet],
                linewidth=1.5,
            )
        plt.legend(loc=2)
        plt.ylabel("control Q")
        plt.xlabel("actuation step")
        plt.tight_layout()
        plt.pause(1.0)
        plt.savefig("saved_figures/control_episode_{}.pdf".format(self.episode_number))
        plt.show()
        plt.pause(2.0)

    def show_drag(self):
        plt.figure()
        crrt_drag = self.history_parameters["drag"].get()
        plt.plot(crrt_drag, label="episode drag", linewidth=1.2)
        plt.plot(
            [0, self.size_history - 1],
            [self.inspection_params["line_drag"], self.inspection_params["line_drag"]],
            label="mean drag no control",
            linewidth=2.5,
            linestyle="--",
        )
        plt.ylabel("measured drag D")
        plt.xlabel("actuation step")
        range_drag_plot = self.inspection_params["range_drag_plot"]
        plt.legend(loc=2)
        plt.ylim(range_drag_plot)
        plt.tight_layout()
        plt.pause(1.0)
        plt.savefig("saved_figures/drag_episode_{}.pdf".format(self.episode_number))
        plt.show()
        plt.pause(2.0)

    def visual_inspection(self):
        total_number_subplots = 5
        crrt_subplot = 1

        if (
            not self.initialized_visualization
            and self.inspection_params["plot"] != False
        ):
            plt.ion()
            plt.subplots(total_number_subplots, 1)
            # ax.set_xlim([0, self.nbr_points_animate_plot])
            # ax.set_ylim([0, 1024])

            self.initialized_visualization = True

        if "plot" in self.inspection_params and self.inspection_params["plot"] != False:
            modulo_base = self.inspection_params["plot"]

            if self.solver_step % modulo_base == 0:

                plt.subplot(total_number_subplots, 1, crrt_subplot)
                plot(self.u_)
                plt.scatter(
                    self.list_positions_probes_x,
                    self.list_positions_probes_y,
                    c="k",
                    marker="o",
                )
                plt.scatter(
                    self.list_positions_jets_x,
                    self.list_positions_jets_y,
                    c="r",
                    marker="o",
                )
                plt.xlim(
                    [
                        -self.geometry_params["front_distance"],
                        self.geometry_params["length"]
                        - self.geometry_params["front_distance"],
                    ]
                )
                plt.ylim(
                    [
                        -self.geometry_params["bottom_distance"],
                        self.geometry_params["width"]
                        - self.geometry_params["bottom_distance"],
                    ]
                )
                plt.ylabel("V")
                crrt_subplot += 1

                plt.subplot(total_number_subplots, 1, crrt_subplot)
                plot(self.p_)
                plt.scatter(
                    self.list_positions_probes_x,
                    self.list_positions_probes_y,
                    c="k",
                    marker="o",
                )
                plt.scatter(
                    self.list_positions_jets_x,
                    self.list_positions_jets_y,
                    c="r",
                    marker="o",
                )
                plt.xlim(
                    [
                        -self.geometry_params["front_distance"],
                        self.geometry_params["length"]
                        - self.geometry_params["front_distance"],
                    ]
                )
                plt.ylim(
                    [
                        -self.geometry_params["bottom_distance"],
                        self.geometry_params["width"]
                        - self.geometry_params["bottom_distance"],
                    ]
                )
                plt.ylabel("P")
                crrt_subplot += 1

                plt.subplot(total_number_subplots, 1, crrt_subplot)
                plt.cla()
                for crrt_jet in range(len(self.geometry_params["jet_positions"])):
                    crrt_jet_data = self.history_parameters[
                        "jet_{}".format(crrt_jet)
                    ].get()
                    plt.plot(crrt_jet_data, label="jet {}".format(crrt_jet))
                plt.legend(loc=6)
                plt.ylabel("M.F.R.")
                crrt_subplot += 1

                # plt.subplot(total_number_subplots, 1, crrt_subplot)
                # plt.cla()
                # for crrt_probe in range(len(self.output_params["locations"])):
                #     if self.output_params["probe_type"] == 'pressure':
                #         crrt_probe_data = self.history_parameters["probe_{}".format(crrt_probe)].get()
                #         plt.plot(crrt_probe_data, label="probe {}".format(crrt_probe))
                #     elif self.output_params["probe_type"] == 'velocity':
                #         crrt_probe_data = self.history_parameters["probe_{}_u".format(crrt_probe)].get()
                #         plt.plot(crrt_probe_data, label="probe {}".format(crrt_probe))
                #         crrt_probe_data = self.history_parameters["probe_{}_v".format(crrt_probe)].get()
                #         plt.plot(crrt_probe_data, label="probe {}".format(crrt_probe))
                # # plt.legend(loc=6)
                # if self.output_params["probe_type"] == "pressure":
                #     plt.ylabel("pressure")
                # elif self.output_params["probe_type"] == "velocity":
                #     plt.ylabel("velocity")
                # if "range_pressure_plot" in self.inspection_params:
                #     range_pressure_plot = self.inspection_params["range_pressure_plot"]
                #     plt.ylim(range_pressure_plot)
                # crrt_subplot += 1

                plt.subplot(total_number_subplots, 1, crrt_subplot)
                ax1 = plt.gca()
                plt.cla()

                crrt_drag = self.history_parameters["drag"].get()

                ax1.plot(crrt_drag, color="r", linestyle="-")
                if "line_drag" in self.inspection_params:
                    ax1.plot(
                        [0, self.size_history - 1],
                        [
                            self.inspection_params["line_drag"],
                            self.inspection_params["line_drag"],
                        ],
                        color="r",
                        linestyle="--",
                    )

                ax1.set_ylabel("drag")
                if "range_drag_plot" in self.inspection_params:
                    range_drag_plot = self.inspection_params["range_drag_plot"]
                    ax1.set_ylim(range_drag_plot)

                ax2 = ax1.twinx()

                crrt_lift = self.history_parameters["lift"].get()

                ax2.plot(crrt_lift, color="b", linestyle="-", label="lift")
                if "line_lift" in self.inspection_params:
                    ax2.plot(
                        [0, self.size_history - 1],
                        [
                            self.inspection_params["line_lift"],
                            self.inspection_params["line_lift"],
                        ],
                        color="b",
                        linestyle="--",
                    )

                ax2.set_ylabel("lift")
                if "range_lift_plot" in self.inspection_params:
                    range_lift_plot = self.inspection_params["range_lift_plot"]
                    ax2.set_ylim(range_lift_plot)

                plt.xlabel("buffer steps")

                crrt_subplot += 1

                plt.subplot(total_number_subplots, 1, crrt_subplot)
                plt.cla()
                crrt_area = self.history_parameters["recirc_area"].get()
                plt.plot(crrt_area)
                plt.ylabel("RecArea")
                plt.xlabel("buffer steps")
                # if "range_drag_plot" in self.inspection_params:
                #    range_drag_plot = self.inspection_params["range_drag_plot"]
                plt.ylim([0, 0.03])
                crrt_subplot += 1

                # plt.tight_layout()
                plt.tight_layout(pad=0, w_pad=0, h_pad=-0.5)
                plt.draw()
                plt.pause(0.5)

        if (
            self.solver_step % self.inspection_params["dump"] == 0
            and self.inspection_params["dump"] < 10000
        ):
            # Affichage en ligne de commande
            print(
                "%s | Ep N: %4d, step: %4d, Rec Area: %.4f, drag: %.4f, lift: %.4f"
                % (
                    self.simu_name,
                    self.episode_number,
                    self.solver_step,
                    self.history_parameters["recirc_area"].get()[-1],
                    self.history_parameters["drag"].get()[-1],
                    self.history_parameters["lift"].get()[-1],
                )
            )

            # Sauvegarde dans un fichier debug de tout ce qui se passe !
            name = "debug.csv"
            if not os.path.exists("saved_models"):
                os.mkdir("saved_models")
            if not os.path.exists("saved_models/" + name):
                with open("saved_models/" + name, "w") as csv_file:
                    spam_writer = csv.writer(
                        csv_file, delimiter=";", lineterminator="\n"
                    )
                    spam_writer.writerow(
                        ["Name", "Episode", "Step", "RecircArea", "Drag", "lift"]
                    )
                    spam_writer.writerow(
                        [
                            self.simu_name,
                            self.episode_number,
                            self.solver_step,
                            self.history_parameters["recirc_area"].get()[-1],
                            self.history_parameters["drag"].get()[-1],
                            self.history_parameters["lift"].get()[-1],
                        ]
                    )
            else:
                with open("saved_models/" + name, "a") as csv_file:
                    spam_writer = csv.writer(
                        csv_file, delimiter=";", lineterminator="\n"
                    )
                    spam_writer.writerow(
                        [
                            self.simu_name,
                            self.episode_number,
                            self.solver_step,
                            self.history_parameters["recirc_area"].get()[-1],
                            self.history_parameters["drag"].get()[-1],
                            self.history_parameters["lift"].get()[-1],
                        ]
                    )

        if (
            "single_run" in self.inspection_params
            and self.inspection_params["single_run"] == True
        ):
            # if ("dump" in self.inspection_params and self.inspection_params["dump"] > 10000):
            self.sing_run_output()

    def sing_run_output(self):
        name = "test_strategy.csv"
        if not os.path.exists("saved_models"):
            os.mkdir("saved_models")
        if not os.path.exists("saved_models/" + name):
            with open("saved_models/" + name, "w") as csv_file:
                spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow(
                    ["Name", "Step", "Drag", "Lift", "RecircArea"]
                    + ["Jet" + str(v) for v in range(len(self.Qs))]
                )
                spam_writer.writerow(
                    [
                        self.simu_name,
                        self.solver_step,
                        self.history_parameters["drag"].get()[-1],
                        self.history_parameters["lift"].get()[-1],
                        self.history_parameters["recirc_area"].get()[-1],
                    ]
                    + [str(v) for v in self.Qs.tolist()]
                )
        else:
            with open("saved_models/" + name, "a") as csv_file:
                spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow(
                    [
                        self.simu_name,
                        self.solver_step,
                        self.history_parameters["drag"].get()[-1],
                        self.history_parameters["lift"].get()[-1],
                        self.history_parameters["recirc_area"].get()[-1],
                    ]
                    + [str(v) for v in self.Qs.tolist()]
                )
        return

    def output_data(self):
        # if "step" in self.inspection_params:
        #     modulo_base = self.inspection_params["step"]

        #     if self.solver_step % modulo_base == 0:
        #         if self.verbose > 0:
        #             print(self.solver_step)
        #             print(self.Qs)
        #             print(self.probes_values)
        #             print(self.drag)
        #             print(self.lift)
        #             print(self.recirc_area)
        #             pass

        if "dump" in self.inspection_params and self.inspection_params["dump"] < 10000:
            modulo_base = self.inspection_params["dump"]

            # Sauvegarde du drag dans le csv a la fin de chaque episode
            self.episode_drags = np.append(
                self.episode_drags, [self.history_parameters["drag"].get()[-1]]
            )
            self.episode_areas = np.append(
                self.episode_areas, [self.history_parameters["recirc_area"].get()[-1]]
            )
            self.episode_lifts = np.append(
                self.episode_lifts, [self.history_parameters["lift"].get()[-1]]
            )

            if (
                self.last_episode_number != self.episode_number
                and "single_run" in self.inspection_params
                and self.inspection_params["single_run"] == False
            ):
                self.last_episode_number = self.episode_number
                avg_drag = np.average(
                    self.episode_drags[len(self.episode_drags) // 2 :]
                )
                avg_area = np.average(
                    self.episode_areas[len(self.episode_areas) // 2 :]
                )
                avg_lift = np.average(
                    self.episode_lifts[len(self.episode_lifts) // 2 :]
                )
                name = "output.csv"
                if not os.path.exists("saved_models"):
                    os.mkdir("saved_models")
                if not os.path.exists("saved_models/" + name):
                    with open("saved_models/" + name, "w") as csv_file:
                        spam_writer = csv.writer(
                            csv_file, delimiter=";", lineterminator="\n"
                        )
                        spam_writer.writerow(
                            ["Episode", "AvgDrag", "AvgLift", "AvgRecircArea"]
                        )
                        spam_writer.writerow(
                            [self.last_episode_number, avg_drag, avg_lift, avg_area]
                        )
                else:
                    with open("saved_models/" + name, "a") as csv_file:
                        spam_writer = csv.writer(
                            csv_file, delimiter=";", lineterminator="\n"
                        )
                        spam_writer.writerow(
                            [self.last_episode_number, avg_drag, avg_lift, avg_area]
                        )
                self.episode_drags = np.array([])
                self.episode_areas = np.array([])
                self.episode_lifts = np.array([])

                if os.path.exists("saved_models/output.csv"):
                    if not os.path.exists("best_model"):
                        shutil.copytree("saved_models", "best_model")

                    else:
                        with open("saved_models/output.csv", "r") as csvfile:
                            data = csv.reader(csvfile, delimiter=";")
                            for row in data:
                                lastrow = row
                            last_iter = lastrow[1]

                        with open("best_model/output.csv", "r") as csvfile:
                            data = csv.reader(csvfile, delimiter=";")
                            for row in data:
                                lastrow = row
                            best_iter = lastrow[1]

                        if float(best_iter) < float(last_iter):
                            print("best_model updated")
                            if os.path.exists("best_model"):
                                shutil.rmtree("best_model")
                            shutil.copytree("saved_models", "best_model")

            # if self.solver_step % modulo_base == 0:
            #     if not self.initialized_output:
            #         self.u_out = File("results/u_out.pvd")
            #         self.p_out = File("results/p_out.pvd")
            #         self.initialized_output = True

            #     if not self.area_probe is None:
            #         self.area_probe.dump(self.area_probe)
            #     self.u_out << self.flow.u_
            #     self.p_out << self.flow.p_

    def __str__(self):
        # printi("Env2DCylinder ---")
        print("")

    def close(self):
        self.ready_to_use = False

    def reset(self):
        if self.solver_step > 0:
            mean_accumulated_drag = self.accumulated_drag / self.solver_step
            mean_accumulated_lift = self.accumulated_lift / self.solver_step

            if self.verbose > -1:
                print(
                    "mean accumulated drag on the whole episode: {}".format(
                        mean_accumulated_drag
                    )
                )

        if self.inspection_params["show_all_at_reset"]:
            self.show_drag()
            self.show_control()

        self.start_class()

        next_state = np.transpose(np.array(self.probes_values))
        if self.noise_probe_index is not None:
            self._set_probe_noise_state(next_state)
        if self.use_selection:
            next_state = next_state[self.index_selection]
        if self.verbose > 0:
            print(next_state)

        self.episode_number += 1

        return next_state

    def _set_probe_noise_state(self, state: np.array):
        state_max = np.max(state)
        state_min = np.min(state)
        for i in self.noise_probe_index:
            state[i] = np.random.uniform(low=state_min, high=state_max)

    def step(self, actions=None):
        action = actions

        if self.verbose > 1:
            print("--- call execute ---")

        if action is None:
            if self.verbose > -1:
                print("carefull, no action given; by default, no jet!")

            nbr_jets = len(self.geometry_params["jet_positions"])
            action = np.zeros((nbr_jets,))

        if self.verbose > 2:
            print(action)

        self.previous_action = self.action
        self.action = action

        # to execute several numerical integration steps
        for crrt_action_nbr in range(self.number_steps_execution):

            # try to force a continuous / smoothe(r) control
            if "smooth_control" in self.optimization_params:
                # printiv(self.optimization_params["smooth_control"])
                # printiv(actions)
                # printiv(self.Qs)
                # self.Qs += self.optimization_params["smooth_control"] * (np.array(action) - self.Qs)  # the solution originally used in the JFM paper
                self.Qs = np.array(self.previous_action) + (
                    np.array(self.action) - np.array(self.previous_action)
                ) / self.number_steps_execution * (
                    crrt_action_nbr + 1
                )  # a linear change in the control
            else:
                self.Qs = np.transpose(np.array(action))

            # impose a zero net Qs
            if "zero_net_Qs" in self.optimization_params:
                if self.optimization_params["zero_net_Qs"]:
                    self.Qs = self.Qs - np.mean(self.Qs)

            # evolve one numerical timestep forward
            self.u_, self.p_ = self.flow.evolve(self.Qs)

            # displaying information that has to do with the solver itself
            self.visual_inspection()
            self.output_data()

            # we have done one solver step
            self.solver_step += 1

            # sample probes and drag
            self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()
            self.drag = self.drag_probe.sample(self.u_, self.p_)
            self.lift = self.lift_probe.sample(self.u_, self.p_)
            self.recirc_area = self.area_probe.sample(self.u_, self.p_)

            # write to the history buffers
            self.write_history_parameters()

            self.accumulated_drag += self.drag
            self.accumulated_lift += self.lift

        # TODO: the next_state may incorporte more information: maybe some time information?
        next_state = np.transpose(np.array(self.probes_values))
        if self.noise_probe_index is not None:
            self._set_probe_noise_state(next_state)
        if self.use_selection:
            next_state = next_state[self.index_selection]

        if self.verbose > 2:
            print(next_state)

        terminal = False

        if self.verbose > 2:
            print(terminal)

        reward = self.compute_reward()

        if self.verbose > 2:
            print(reward)

        if self.verbose > 1:
            print("--- done execute ---")

        return (next_state, terminal, reward)

        # return area

    def compute_reward(self):
        # NOTE: reward should be computed over the whole number of iterations in each execute loop
        if (
            self.reward_function == "plain_drag"
        ):  # a bit dangerous, may be injecting some momentum
            values_drag_in_last_execute = self.history_parameters["drag"].get()[
                -self.number_steps_execution :
            ]
            return (
                np.mean(values_drag_in_last_execute) + self.drag_constant
            )  # TODO: the 0.159 value is a proxy value corresponding to the mean drag when no control; may depend on the geometry
        elif self.reward_function == "recirculation_area":
            return -self.area_probe.sample(self.u_, self.p_)
        elif self.reward_function == "max_recirculation_area":
            return self.area_probe.sample(self.u_, self.p_)
        elif (
            self.reward_function == "drag"
        ):  # a bit dangerous, may be injecting some momentum
            return self.history_parameters["drag"].get()[-1] + self.drag_constant
        elif (
            self.reward_function == "drag_plain_lift"
        ):  # a bit dangerous, may be injecting some momentum
            avg_length = min(500, self.number_steps_execution)
            avg_drag = np.mean(self.history_parameters["drag"].get()[-avg_length:])
            avg_lift = np.mean(self.history_parameters["lift"].get()[-avg_length:])
            r = avg_drag + self.drag_constant - self.lift_penalize * abs(avg_lift)
            return 1 * r
        elif (
            self.reward_function == "max_plain_drag"
        ):  # a bit dangerous, may be injecting some momentum
            values_drag_in_last_execute = self.history_parameters["drag"].get()[
                -self.number_steps_execution :
            ]
            return -(np.mean(values_drag_in_last_execute) + self.drag_constant)
        elif (
            self.reward_function == "drag_avg_abs_lift"
        ):  # a bit dangerous, may be injecting some momentum
            avg_length = min(500, self.number_steps_execution)
            avg_abs_lift = np.mean(
                np.absolute(self.history_parameters["lift"].get()[-avg_length:])
            )
            avg_drag = np.mean(self.history_parameters["drag"].get()[-avg_length:])
            return avg_drag + self.drag_constant - self.lift_penalize * avg_abs_lift

        # TODO: implement some reward functions that take into account how much energy / momentum we inject into the flow

        else:
            raise RuntimeError(
                "reward function {} not yet implemented".format(self.reward_function)
            )

    def states(self):
        if self.output_params["probe_type"] == "pressure":
            return dict(
                type="float",
                shape=(
                    len(self.output_params["locations"])
                    * self.optimization_params["num_steps_in_pressure_history"],
                ),
            )

        elif self.output_params["probe_type"] == "velocity":
            return dict(
                type="float",
                shape=(
                    2
                    * len(self.output_params["locations"])
                    * self.optimization_params["num_steps_in_pressure_history"],
                ),
            )

    def actions(self):
        # NOTE: we could also have several levels of dict in dict, for example:
        # return { str(i): dict(continuous=True, min_value=0, max_value=1) for i in range(self.n + 1) }

        return dict(
            type="float",
            shape=(len(self.geometry_params["jet_positions"]),),
            min_value=self.optimization_params["min_value_jet_MFR"],
            max_value=self.optimization_params["max_value_jet_MFR"],
        )

    def max_episode_timesteps(self):
        return None
