import os
from typing import Dict, List, Literal, Optional, Tuple

import numpy
from openff.utilities import temporary_cd
from openmm import unit
from pymbar import MBAR, timeseries

from absolv.models import EquilibriumProtocol, TransferFreeEnergySchema
from absolv.runners._runners import BaseRunner
from absolv.utilities.openmm import OpenMMPlatform


class EquilibriumRunner(BaseRunner):
    """A utility class for setting up, running, and analyzing equilibrium
    free energy calculations.
    """

    @classmethod
    def run(
        cls,
        schema: TransferFreeEnergySchema,
        directory: str = "absolv-experiment",
        platform: OpenMMPlatform = "CUDA",
        states: Optional[
            Dict[Literal["solvent-a", "solvent-b"], Optional[List[int]]]
        ] = None,
    ):
        """Perform a simulation at each lambda window and for each solvent.

        Notes:
            This method assumes ``setup`` as already been run.

        Args:
            schema: The schema defining the calculation to perform.
            directory: The directory containing the input files.
            platform: The OpenMM platform to run using.
            states: An optional dictionary of the specific solvent and states to run.
                The dictionary should have the form
                ``{"solvent-a": [state_index_0, ..], "solvent-b": [state_index_0, ..]}``.
                All lambda windows for a solvent can be run by specifying ``None``, e.g.
                ``{"solvent-a": None, "solvent-b": [0, 1, 2]}``
        """

        states = (
            states if states is not None else {"solvent-a": None, "solvent-b": None}
        )

        for solvent_index, protocol in zip(
            states, (schema.alchemical_protocol_a, schema.alchemical_protocol_b)
        ):

            with temporary_cd(os.path.join(directory, solvent_index)):

                cls._run_solvent(
                    protocol, schema.state, platform, states[solvent_index]
                )

    @classmethod
    def _analyze_solvent(
        cls,
        protocol: EquilibriumProtocol,
    ) -> Tuple[unit.Quantity, unit.Quantity]:

        n_iterations = protocol.production_protocol.n_iterations
        n_states = protocol.n_states

        u_kln = numpy.zeros([n_states, n_states, n_iterations], numpy.float64)

        for state_index in range(n_states):

            state_potentials = numpy.genfromtxt(
                os.path.join(
                    f"state-{state_index}",
                    "lambda-potentials.csv",
                ),
                delimiter=" ",
            )
            state_potentials = state_potentials.reshape((n_iterations, n_states))

            u_kln[state_index] = state_potentials.T

        n_k = numpy.zeros([n_states], numpy.int32)

        for k in range(n_states):

            _, g, _ = timeseries.detectEquilibration(u_kln[k, k, :])
            indices = timeseries.subsampleCorrelatedData(u_kln[k, k, :], g=g)

            n_k[k] = len(indices)
            u_kln[k, :, 0 : n_k[k]] = u_kln[k, :, indices].T

        # Compute free energy differences and statistical uncertainties
        mbar = MBAR(u_kln, n_k)

        delta_f_ij, delta_delta_f_ij = mbar.getFreeEnergyDifferences()

        return delta_f_ij[0, -1], delta_delta_f_ij[0, -1]

    @classmethod
    def analyze(
        cls,
        schema: TransferFreeEnergySchema,
        directory: str = "absolv-experiment",
    ):
        """Analyze the outputs of the simulations to compute the transfer free energy
        using MBAR.

        Notes:
            This method assumes ``setup`` and ``run`` have already been successfully run.

        Args:
            schema: The schema defining the calculation to perform.
            directory: The directory containing the input and simulation files.
        """

        free_energies = {}

        for solvent_index, protocol in zip(
            ("a", "b"), (schema.alchemical_protocol_a, schema.alchemical_protocol_b)
        ):

            with temporary_cd(os.path.join(directory, f"solvent-{solvent_index}")):

                value, std_error = cls._analyze_solvent(protocol)

                free_energies[f"solvent-{solvent_index}"] = {
                    "value": value,
                    "std_error": std_error,
                }

        free_energies["a->b"] = {
            "value": free_energies["solvent-b"]["value"]
            - free_energies["solvent-a"]["value"],
            "std_error": numpy.sqrt(
                free_energies["solvent-a"]["std_error"] ** 2
                + free_energies["solvent-b"]["std_error"] ** 2
            ),
        }

        return free_energies
