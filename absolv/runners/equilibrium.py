import os
import pickle
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy
from openff.toolkit.topology import Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.utilities import temporary_cd
from openmm import openmm, unit
from pymbar import MBAR, timeseries
from tqdm import tqdm

from absolv.factories.alchemical import OpenMMAlchemicalFactory
from absolv.factories.coordinate import PACKMOLCoordinateFactory
from absolv.models import EquilibriumProtocol, State, TransferFreeEnergySchema
from absolv.simulations import AlchemicalOpenMMSimulation
from absolv.utilities.openmm import OpenMMPlatform
from absolv.utilities.topology import topology_to_atom_indices

SystemGenerator = Callable[[Topology, unit.Quantity], openmm.System]


class EquilibriumRunner:
    """A utility class for setting up, running, and analyzing equilibrium
    free energy calculations.
    """

    @classmethod
    def _setup_solvent(
        cls,
        components: List[Tuple[str, int]],
        force_field: Union[ForceField, SystemGenerator],
        n_solute_molecules: int,
        n_solvent_molecules: int,
    ):
        """Creates the input files for a particular solvent phase.

        Args:
            components: A list of the form ``components[i] = (smiles_i, count_i)`` where
                ``smiles_i`` is the SMILES representation of component `i` and
                ``count_i`` is the number of corresponding instances of that component
                to create.

                It is expected that the first ``n_solute_molecules`` entries in this
                list correspond to molecules that will be alchemically transformed,
                and the remaining ``n_solvent_molecules`` entries correspond to molecules
                that will not.
            force_field: The force field, or a callable that transforms an OpenFF
                topology and a set of coordinates into an OpenMM system **without**
                any alchemical modifications, to run the calculations using.
            n_solute_molecules: The number of solute molecule.
            n_solvent_molecules: The number of solvent molecule.
        """

        is_vacuum = n_solvent_molecules == 0

        topology, coordinates = PACKMOLCoordinateFactory.generate(components)
        topology.box_vectors = None if is_vacuum else topology.box_vectors

        atom_indices = topology_to_atom_indices(topology)

        alchemical_indices = atom_indices[:n_solute_molecules]
        persistent_indices = atom_indices[n_solute_molecules:]

        if isinstance(force_field, ForceField):
            original_system = force_field.create_openmm_system(topology)
        else:
            original_system: openmm.System = force_field(topology, coordinates)

        alchemical_system = OpenMMAlchemicalFactory.generate(
            original_system, alchemical_indices, persistent_indices
        )

        topology.to_file("coords-initial.pdb", coordinates)
        numpy.save("coords-initial.npy", coordinates.value_in_unit(unit.angstrom))

        with open("system-chemical.xml", "w") as file:
            file.write(openmm.XmlSerializer.serializeSystem(original_system))

        with open("system-alchemical.xml", "w") as file:
            file.write(openmm.XmlSerializer.serializeSystem(alchemical_system))

        with open("topology.pkl", "wb") as file:
            pickle.dump(topology, file)

    @classmethod
    def setup(
        cls,
        schema: TransferFreeEnergySchema,
        force_field: Union[ForceField, SystemGenerator],
        directory: str = "absolv-experiment",
    ):
        """Prepare the input files needed to compute the free energy and store them
        in the specified directory.

        Args:
            schema: The schema defining the calculation to perform.
            force_field: The force field, or a callable that transforms an OpenFF
                topology and a set of coordinates into an OpenMM system **without**
                any alchemical modifications, to run the calculations using.
            force_field: The force field (or system generator) to use to generate
                the chemical system object.
            directory: The directory to create the input files in.
        """

        n_solute_molecules = schema.system.n_solute_molecules

        for solvent_index, components, n_solvent_molecules in zip(
            ("a", "b"),
            schema.system.to_components(),
            (schema.system.n_solvent_molecules_a, schema.system.n_solvent_molecules_b),
        ):

            solvent_directory = os.path.join(directory, f"solvent-{solvent_index}")
            os.makedirs(solvent_directory, exist_ok=False)

            with temporary_cd(solvent_directory):

                cls._setup_solvent(
                    components, force_field, n_solute_molecules, n_solvent_molecules
                )

    @classmethod
    def _run_solvent(
        cls,
        protocol: EquilibriumProtocol,
        state: State,
        platform: OpenMMPlatform,
        states: Optional[List[int]] = None,
    ):

        states = states if states is not None else list(range(protocol.n_states))

        with open("system-alchemical.xml", "r") as file:
            alchemical_system = openmm.XmlSerializer.deserializeSystem(file.read())

        with open("topology.pkl", "rb") as file:
            topology = pickle.load(file)

        coordinates = numpy.load("coords-initial.npy") * unit.angstrom

        for state_index in tqdm(states, desc="state", ncols=80):

            simulation = AlchemicalOpenMMSimulation(
                alchemical_system,
                coordinates,
                topology.box_vectors,
                State(
                    temperature=state.temperature,
                    pressure=None if topology.box_vectors is None else state.pressure,
                ),
                protocol,
                state_index,
                platform if topology.box_vectors is not None else "Reference",
            )

            simulation.run(os.path.join(f"state-{state_index}"))

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
