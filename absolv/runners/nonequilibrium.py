import os
from typing import Callable

import mdtraj
import numpy
import pymbar
from openff.toolkit.topology import Topology
from openff.utilities import temporary_cd
from openmm import openmm, unit
from tqdm import tqdm

from absolv.models import (
    DeltaG,
    EquilibriumProtocol,
    NonEquilibriumProtocol,
    State,
    TransferFreeEnergyResult,
    TransferFreeEnergySchema,
)
from absolv.runners._runners import BaseRunner
from absolv.simulations import NonEquilibriumOpenMMSimulation
from absolv.utilities.openmm import OpenMMPlatform

SystemGenerator = Callable[[Topology, unit.Quantity], openmm.System]


class NonEquilibriumRunner(BaseRunner):
    """A utility class for setting up, running, and analyzing non-equilibrium
    free energy calculations.
    """

    @classmethod
    def _run_switching(
        cls,
        protocol: NonEquilibriumProtocol,
        state: State,
        platform: OpenMMPlatform,
    ):

        topology, coordinates, _, alchemical_system = cls._load_solvent_inputs("")

        mdtraj_topology = mdtraj.Topology.from_openmm(topology.to_openmm())

        trajectory_0 = mdtraj.load_dcd(
            os.path.join("state-0", "production-trajectory.dcd"), mdtraj_topology
        )
        trajectory_1 = mdtraj.load_dcd(
            os.path.join("state-1", "production-trajectory.dcd"), mdtraj_topology
        )

        assert len(trajectory_0) == len(
            trajectory_1
        ), "trajectories ran in the two end states must have the same length"

        if os.path.isfile("forward-work.csv") and os.path.isfile("reverse-work.csv"):

            forward_work = numpy.genfromtxt("forward-work.csv", delimiter=" ")
            reverse_work = numpy.genfromtxt("reverse-work.csv", delimiter=" ")

            return forward_work, reverse_work

        forward_work = numpy.zeros(len(trajectory_0))
        reverse_work = numpy.zeros(len(trajectory_0))

        for frame_index in tqdm(range(len(trajectory_0)), desc=" NEQ frame", ncols=80):

            coordinates_0 = trajectory_0.xyz[frame_index] * unit.nanometers
            box_vectors_0 = (
                (trajectory_0.unitcell_vectors[frame_index] * unit.nanometers)
                if trajectory_0.unitcell_vectors is not None
                else None
            )

            coordinates_1 = trajectory_1.xyz[frame_index] * unit.nanometers
            box_vectors_1 = (
                (trajectory_1.unitcell_vectors[frame_index] * unit.nanometers)
                if trajectory_1.unitcell_vectors is not None
                else None
            )

            simulation = NonEquilibriumOpenMMSimulation(
                alchemical_system,
                State(
                    temperature=state.temperature,
                    pressure=None if box_vectors_0 is None else state.pressure,
                ),
                coordinates_0,
                box_vectors_0,
                coordinates_1,
                box_vectors_1,
                protocol.switching_protocol,
                platform,
            )

            forward_work[frame_index], reverse_work[frame_index] = simulation.run()

        numpy.savetxt("forward-work.csv", forward_work, delimiter=" ")
        numpy.savetxt("reverse-work.csv", reverse_work, delimiter=" ")

    @classmethod
    def run(
        cls,
        directory: str = "absolv-experiment",
        platform: OpenMMPlatform = "CUDA",
    ):
        """Performs an **equilibrium** simulation at the two end states (i.e. fully
        interaction and fully de-coupled solute) for each solvent followed by
        non-equilibrium switching simulations between each end states to compute the
        forward and reverse work values.

        Notes:
            This method assumes ``setup`` as already been run.

        Args:
            directory: The directory containing the input files.
            platform: The OpenMM platform to run using.
        """

        schema = TransferFreeEnergySchema.parse_file(
            os.path.join(directory, "schema.json")
        )

        for solvent_index, protocol in zip(
            ("solvent-a", "solvent-b"),
            (schema.alchemical_protocol_a, schema.alchemical_protocol_b),
        ):

            assert isinstance(protocol, NonEquilibriumProtocol)

            with temporary_cd(os.path.join(directory, solvent_index)):

                cls._run_solvent(
                    EquilibriumProtocol(
                        minimization_protocol=protocol.minimization_protocol,
                        equilibration_protocol=protocol.equilibration_protocol,
                        production_protocol=protocol.production_protocol,
                        lambda_sterics=[1.0, 0.0],
                        lambda_electrostatics=[1.0, 0.0],
                    ),
                    schema.state,
                    platform,
                )

                cls._run_switching(protocol, schema.state, platform)

    @classmethod
    def analyze(
        cls,
        directory: str = "absolv-experiment",
    ):
        """Analyze the outputs of the non-equilibrium simulations to compute the transfer
        free energy using the Crooks relation.

        Notes:
            This method assumes ``setup`` and ``run`` have already been successfully run.

        Args:
            directory: The directory containing the input and simulation files.
        """

        schema = TransferFreeEnergySchema.parse_file(
            os.path.join(directory, "schema.json")
        )

        free_energies = {}

        for solvent_index in ("solvent-a", "solvent-b"):

            forward_work = numpy.genfromtxt(
                os.path.join(directory, solvent_index, "forward-work.csv"),
                delimiter=" ",
            )
            reverse_work = numpy.genfromtxt(
                os.path.join(directory, solvent_index, "reverse-work.csv"),
                delimiter=" ",
            )

            value, std_error = pymbar.BAR(forward_work, reverse_work)

            free_energies[solvent_index] = {"value": value, "std_error": std_error}

        return TransferFreeEnergyResult(
            input_schema=schema,
            delta_g_solvent_a=DeltaG(
                value=free_energies["solvent-a"]["value"],
                std_error=free_energies["solvent-a"]["std_error"],
            ),
            delta_g_solvent_b=DeltaG(
                value=free_energies["solvent-b"]["value"],
                std_error=free_energies["solvent-b"]["std_error"],
            ),
        )
