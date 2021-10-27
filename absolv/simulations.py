import os.path
from typing import IO, Optional

import openmm
from openff.utilities import temporary_cd
from openmm import unit
from openmm.app import DCDFile
from tqdm import tqdm

from absolv.models import EquilibriumProtocol, SimulationProtocol, State
from absolv.utilities.openmm import (
    OpenMMPlatform,
    build_context,
    minimize,
    set_alchemical_lambdas,
)


class _OpenMMTopology:
    """A fake OpenMM topology class that allows us to use the OpenMM DCD writer
    without needing to know the full topological information.
    """

    def __init__(self, n_positions: int, box_vectors):
        self._atoms = list(range(n_positions))
        self._box_vectors = box_vectors

    def atoms(self):
        return self._atoms

    def getPeriodicBoxVectors(self):
        return self._box_vectors

    def getUnitCellDimensions(self):
        return (
            None
            if self._box_vectors is None
            else [self._box_vectors[i][i] for i in range(3)]
        )


class EquilibriumOpenMMSimulation:
    """A class that simplifies the process of running an equilibrium simulation with
    OpenMM, including performing energy minimizations, equilibration steps, and
    checkpointing.
    """

    def __init__(
        self,
        system: openmm.System,
        positions: unit.Quantity,
        box_vectors: Optional[unit.Quantity],
        state: State,
        protocol: EquilibriumProtocol,
        lambda_index: int,
        platform: OpenMMPlatform,
    ):
        """

        Args:
            system: The OpenMM system to simulate.
            positions: The initial positions of all atoms.
            box_vectors: The (optional) initial periodic box vectors.
            state: The state to simulate at.
            protocol: The protocol to simulate according to.
            lambda_index: The index defining which value of lambda to simulate at.
            platform: The OpenMM platform to simulate using.
        """

        self._context = build_context(
            system,
            positions,
            box_vectors,
            state.temperature * unit.kelvin,
            None if state.pressure is None else state.pressure * unit.atmosphere,
            platform=platform,
        )
        self._mock_topology = _OpenMMTopology(system.getNumParticles(), box_vectors)

        self._protocol = protocol

        self._lambda_sterics = protocol.lambda_sterics[lambda_index]
        self._lambda_electrostatics = protocol.lambda_electrostatics[lambda_index]

        set_alchemical_lambdas(
            self._context, self._lambda_sterics, self._lambda_electrostatics
        )

    def _restore_state(self, name: str) -> bool:
        """Restores the state of a context from a serialized state file.

        Args:
            name: The name of the state. The associate state file should be named
                ``{name}-state.xml``.

        Returns:
            Whether the state was successfully restored or not.
        """

        if not os.path.isfile(f"{name}-state.xml"):
            return False

        with open(f"{name}-state.xml") as file:
            state = openmm.XmlSerializer.deserialize(file.read())

        self._context.setState(state)
        return True

    def _save_state(self, name: str):
        """Save the state of a context to a serialized state file.

        Args:
            name: The name of the state. The associate state file will be named
                ``{name}-state.xml``.
        """
        state: openmm.State = self._context.getState(
            getPositions=True,
            getVelocities=True,
            getForces=True,
            getParameters=True,
            getIntegratorParameters=True,
        )

        with open(f"{name}-state.xml", "w") as file:
            file.write(openmm.XmlSerializer.serialize(state))

    def _minimize(self):
        """Energy minimize the context if minimization has not already occurred."""

        if self._protocol.minimization_protocol is None:
            return

        if self._restore_state("minimized"):
            return

        minimize(
            self._context,
            self._protocol.minimization_protocol.tolerance
            * unit.kilojoules_per_mole
            / unit.nanometer,
            self._protocol.minimization_protocol.max_iterations,
        )

        self._save_state("minimized")

    def _begin_iteration(self, iteration: int, name: str):
        """A hook called before a simulation iteration is about to begin."""

    def _end_iteration(self, iteration: int, name: str):
        """A hook called after a simulation iteration has just finished."""

    def _simulate(self, protocol: SimulationProtocol, name: str):
        """Evolve the state of the context according to a specific protocol.

        Args:
            protocol: The protocol to simulate according to.
            name: The name associated with this simulation.
        """

        if self._restore_state(f"{name}-final"):
            return

        integrator: openmm.LangevinIntegrator = self._context.getIntegrator()
        integrator.setStepSize(protocol.timestep * unit.femtoseconds)
        integrator.setFriction(protocol.thermostat_friction / unit.picoseconds)

        has_restored = self._restore_state(f"{name}-chk")

        trajectory_mode = "r+b" if os.path.isfile(f"{name}-trajectory.dcd") else "wb"

        with open(f"{name}-trajectory.dcd", trajectory_mode) as dcd_stream:

            dcd_file = DCDFile(
                dcd_stream,
                self._mock_topology,
                integrator.getStepSize(),
                0,
                0,
                has_restored,
            )

            completed_n_iterations = dcd_file._modelCount

            for iteration in tqdm(
                range(protocol.n_iterations - completed_n_iterations),
                desc=f" {name}",
                ncols=80,
            ):

                self._begin_iteration(completed_n_iterations + iteration, name)

                integrator = self._context.getIntegrator()
                integrator.step(protocol.n_steps_per_iteration)

                self._save_state(f"{name}-chk")

                state: openmm.State = self._context.getState(getPositions=True)

                dcd_file.writeModel(
                    state.getPositions(),
                    periodicBoxVectors=state.getPeriodicBoxVectors(),
                )

                self._end_iteration(completed_n_iterations + iteration, name)

        self._save_state(f"{name}-final")

        try:
            os.remove(f"{name}-chk-state.xml")
        except OSError:
            pass

    def run(self, directory: Optional[str]):
        """Run the full simulation, restarting from where it left off if a previous
        attempt to run had already been made.

        Args:
            directory: The (optional) directory to run in. If no directory is specified
                the outputs will be stored in a temporary directory and no restarts will
                be possible.
        """

        if directory is not None and len(directory) > 0:
            os.makedirs(directory, exist_ok=True)

        with temporary_cd(directory):

            self._minimize()

            if self._protocol.equilibration_protocol is not None:
                self._simulate(self._protocol.equilibration_protocol, "equilibration")

            if self._protocol.production_protocol is not None:
                self._simulate(self._protocol.production_protocol, "production")


class AlchemicalOpenMMSimulation(EquilibriumOpenMMSimulation):
    """An extension to the base equilibrium simulation class that will additionally
    compute the reduced potentials at each lambda value after each production simulation
    iteration and save those values to disk.
    """

    def __init__(
        self,
        system: openmm.System,
        positions: unit.Quantity,
        box_vectors: Optional[unit.Quantity],
        state: State,
        protocol: EquilibriumProtocol,
        lambda_index: int,
        platform: OpenMMPlatform,
    ):

        super(AlchemicalOpenMMSimulation, self).__init__(
            system, positions, box_vectors, state, protocol, lambda_index, platform
        )

        self._energies_file: Optional[IO] = None

        self._beta = 1.0 / (
            unit.BOLTZMANN_CONSTANT_kB * (state.temperature * unit.kelvin)
        )
        self._pressure = (
            None if state.pressure is None else state.pressure * unit.atmosphere
        )

    def _compute_reduced_potential(self) -> float:
        """Computes the reduced potential of the contexts current state.

        Returns:
            The reduced potential.
        """

        state = self._context.getState(getEnergy=True)

        unreduced_potential = state.getPotentialEnergy() / unit.AVOGADRO_CONSTANT_NA

        if self._pressure is not None:
            unreduced_potential += self._pressure * state.getPeriodicBoxVolume()

        return unreduced_potential * self._beta

    def _begin_iteration(self, iteration: int, name: str):
        super(AlchemicalOpenMMSimulation, self)._begin_iteration(iteration, name)

        if name == "production" and self._energies_file is None:
            self._energies_file = open("lambda-potentials.csv", "a")

    def _end_iteration(self, iteration: int, name: str):
        super(AlchemicalOpenMMSimulation, self)._end_iteration(iteration, name)

        if name != "production":
            return

        alchemical_potentials = []

        for lambda_sterics, lambda_electrostatics in zip(
            self._protocol.lambda_sterics, self._protocol.lambda_electrostatics
        ):

            set_alchemical_lambdas(self._context, lambda_sterics, lambda_electrostatics)
            alchemical_potentials.append(self._compute_reduced_potential())

        # Restore the lambda values.
        set_alchemical_lambdas(
            self._context, self._lambda_sterics, self._lambda_electrostatics
        )

        self._energies_file.write(
            " ".join(f"{energy:+.10f}" for energy in alchemical_potentials) + "\n"
        )
        self._energies_file.flush()
