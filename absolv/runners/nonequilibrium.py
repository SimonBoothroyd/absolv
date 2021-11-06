import os
from typing import Callable

import mdtraj
import numpy
from openff.toolkit.topology import Topology
from openff.utilities import temporary_cd
from openmm import openmm, unit
from tqdm import tqdm

from absolv.models import (
    EquilibriumProtocol,
    NonEquilibriumProtocol,
    State,
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
    def run_equilibrium(
        cls,
        schema: TransferFreeEnergySchema,
        directory: str = "absolv-experiment",
        platform: OpenMMPlatform = "CUDA",
    ):
        """Perform an **equilibrium** simulation at the two end states (i.e. fully
        interaction and fully de-coupled solute) for each solvent.

        Notes:
            This method assumes ``setup`` as already been run.

        Args:
            schema: The schema defining the calculation to perform.
            directory: The directory containing the input files.
            platform: The OpenMM platform to run using.
        """

        solvent_indices = ["solvent-a", "solvent-b"]

        for solvent_index, protocol in zip(
            solvent_indices,
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

        forward_work = numpy.zeros(len(trajectory_0))
        reverse_work = numpy.zeros(len(trajectory_0))

        for frame_index in tqdm(
            range(len(trajectory_0)),
            desc=f" NEQ frame",
            ncols=80,
        ):

            coordinates_0 = trajectory_0.xyz[frame_index] * unit.nanometers
            box_vectors_0 = trajectory_0.unitcell_vectors[frame_index] * unit.nanometers

            coordinates_1 = trajectory_1.xyz[frame_index] * unit.nanometers
            box_vectors_1 = trajectory_1.unitcell_vectors[frame_index] * unit.nanometers

            simulation = NonEquilibriumOpenMMSimulation(
                alchemical_system,
                state,
                coordinates_0,
                box_vectors_0,
                coordinates_1,
                box_vectors_1,
                protocol.switching_protocol,
                platform,
            )

            forward_work[frame_index], reverse_work[frame_index] = simulation.run(
                os.path.join(f"neq-frame-{frame_index}")
            )

        numpy.savetxt(
            "forward-work.csv", forward_work, delimiter=" "
        )
        numpy.savetxt(
            "reverse-work.csv", reverse_work, delimiter=" "
        )

    @classmethod
    def run_switching(
        cls,
        schema: TransferFreeEnergySchema,
        directory: str = "absolv-experiment",
        platform: OpenMMPlatform = "CUDA",
    ):
        """Perform an **equilibrium** simulation at the two end states (i.e. fully
        interaction and fully de-coupled solute) for each solvent.

        Notes:
            This method assumes ``setup`` as already been run.

        Args:
            schema: The schema defining the calculation to perform.
            directory: The directory containing the input files.
            platform: The OpenMM platform to run using.
        """

        solvent_indices = ["solvent-a", "solvent-b"]

        for solvent_index, protocol in zip(
            solvent_indices,
            (schema.alchemical_protocol_a, schema.alchemical_protocol_b),
        ):

            with temporary_cd(os.path.join(directory, solvent_index)):
                cls._run_switching(protocol, schema.state, platform)
