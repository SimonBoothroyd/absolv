import os
from typing import Callable

from openff.toolkit.topology import Topology
from openff.utilities import temporary_cd
from openmm import openmm, unit

from absolv.models import (
    EquilibriumProtocol,
    NonEquilibriumProtocol,
    TransferFreeEnergySchema,
)
from absolv.runners._runners import BaseRunner
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

        states = ["solvent-a", "solvent-b"]

        for solvent_index, protocol in zip(
            states, (schema.alchemical_protocol_a, schema.alchemical_protocol_b)
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
