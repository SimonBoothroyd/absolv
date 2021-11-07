import os.path
from typing import IO, Optional, Tuple

import numpy
import openmm
from openff.utilities import temporary_cd
from openmm import unit
from openmm.app import DCDFile
from tqdm import tqdm

from absolv.models import (
    EquilibriumProtocol,
    SimulationProtocol,
    State,
    SwitchingProtocol,
)
from absolv.utilities.openmm import (
    OpenMMPlatform,
    build_context,
    minimize,
    set_alchemical_lambdas,
    set_coordinates,
)


class _OpenMMTopology:
    """A fake OpenMM topology class that allows us to use the OpenMM DCD writer
    without needing to know the full topological information.
    """

    def __init__(self, n_coordinates: int, box_vectors):
        self._atoms = list(range(n_coordinates))
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


class _BaseOpenMMSimulation:
    def __init__(
        self,
        system: openmm.System,
        coordinates: unit.Quantity,
        box_vectors: Optional[unit.Quantity],
        state: State,
        platform: OpenMMPlatform,
    ):
        """

        Args:
            system: The OpenMM system to simulate.
            coordinates: The initial coordinates of all atoms.
            box_vectors: The (optional) initial periodic box vectors.
            state: The state to simulate at.
            platform: The OpenMM platform to simulate using.
        """

        self._context = build_context(
            system,
            coordinates,
            box_vectors,
            state.temperature * unit.kelvin,
            None if state.pressure is None else state.pressure * unit.atmosphere,
            platform=platform,
        )
        self._mock_topology = _OpenMMTopology(system.getNumParticles(), box_vectors)

        self._beta = 1.0 / (
            unit.BOLTZMANN_CONSTANT_kB * (state.temperature * unit.kelvin)
        )
        self._pressure = (
            (state.pressure * unit.atmosphere) if state.pressure is not None else None
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


class EquilibriumOpenMMSimulation(_BaseOpenMMSimulation):
    """A class that simplifies the process of running an equilibrium simulation with
    OpenMM, including performing energy minimizations, equilibration steps, and
    checkpointing.
    """

    def __init__(
        self,
        system: openmm.System,
        coordinates: unit.Quantity,
        box_vectors: Optional[unit.Quantity],
        state: State,
        protocol: EquilibriumProtocol,
        lambda_index: int,
        platform: OpenMMPlatform,
    ):
        """

        Args:
            system: The OpenMM system to simulate.
            coordinates: The initial coordinates of all atoms.
            box_vectors: The (optional) initial periodic box vectors.
            state: The state to simulate at.
            protocol: The protocol to simulate according to.
            lambda_index: The index defining which value of lambda to simulate at.
            platform: The OpenMM platform to simulate using.
        """

        super(EquilibriumOpenMMSimulation, self).__init__(
            system, coordinates, box_vectors, state, platform
        )

        self._protocol = protocol

        self._lambda_sterics = protocol.lambda_sterics[lambda_index]
        self._lambda_electrostatics = protocol.lambda_electrostatics[lambda_index]

        set_alchemical_lambdas(
            self._context, self._lambda_sterics, self._lambda_electrostatics
        )

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
        coordinates: unit.Quantity,
        box_vectors: Optional[unit.Quantity],
        state: State,
        protocol: EquilibriumProtocol,
        lambda_index: int,
        platform: OpenMMPlatform,
    ):

        super(AlchemicalOpenMMSimulation, self).__init__(
            system, coordinates, box_vectors, state, protocol, lambda_index, platform
        )

        self._energies_file: Optional[IO] = None

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


class NonEquilibriumOpenMMSimulation(_BaseOpenMMSimulation):
    """A class that simplifies the process of running a non-equilibrium simulation with
    OpenMM, whereby a system is non-reversibly pulled along an alchemical pathway
    as described by Ballard and Jarzynski [1] (Figure 3) and Gapsys et al [2].

    Both the forward and reverse directions will be simulated by this class.

    References:
        [1] Ballard, Andrew J., and Christopher Jarzynski. "Replica exchange with
        nonequilibrium switches: Enhancing equilibrium sampling by increasing replica
        overlap." The Journal of chemical physics 136.19 (2012): 194101.

        [2] Gapsys, Vytautas, et al. "Large scale relative protein ligand binding
        affinities using non-equilibrium alchemy." Chemical Science 11.4 (2020):
        1140-1152.
    """

    def __init__(
        self,
        system: openmm.System,
        state: State,
        coordinates_0: unit.Quantity,
        box_vectors_0: Optional[unit.Quantity],
        coordinates_1: unit.Quantity,
        box_vectors_1: Optional[unit.Quantity],
        protocol: SwitchingProtocol,
        platform: OpenMMPlatform,
    ):
        """

        Args:
            system: The OpenMM system to simulate.
            state: The state to simulate at.
            coordinates_0:
            box_vectors_0:
            coordinates_1:
            box_vectors_1:
            protocol:
            platform: The OpenMM platform to simulate using.
        """

        super(NonEquilibriumOpenMMSimulation, self).__init__(
            system, coordinates_0, box_vectors_0, state, platform
        )

        integrator: openmm.LangevinIntegrator = self._context.getIntegrator()
        integrator.setStepSize(protocol.timestep * unit.femtoseconds)
        integrator.setFriction(protocol.thermostat_friction / unit.picoseconds)

        self._protocol = protocol

        self._state_0 = (coordinates_0, box_vectors_0)
        self._state_1 = (coordinates_1, box_vectors_1)

    def _compute_lambdas(
        self, time_frame: int, reverse_direction: bool
    ) -> Tuple[float, float, float]:
        """Computes the values of the global, electrostatics and sterics lambdas for
        a given time frame.

        Args:
            time_frame: The current time frame (i.e. current time / timestep).
            reverse_direction: Whether to move from state 1 -> 0 rather than 0 -> 1.

        Returns:
            The values of the global, electrostatics and sterics lambdas
        """

        n_electrostatic_timesteps = (
            self._protocol.n_electrostatic_steps
            * self._protocol.n_steps_per_electrostatic_step
        )
        n_steric_timesteps = (
            self._protocol.n_steric_steps * self._protocol.n_steps_per_steric_step
        )

        n_total_timesteps = n_electrostatic_timesteps + n_steric_timesteps

        if reverse_direction:
            time_frame = n_total_timesteps - time_frame

        time = time_frame * self._protocol.timestep

        time_electrostatics = self._protocol.timestep * n_electrostatic_timesteps
        time_total = self._protocol.timestep * n_total_timesteps

        lambda_global = (time_total - time) / time_total

        lambda_electrostatics = (
            0.0
            if time_frame >= n_electrostatic_timesteps or n_electrostatic_timesteps == 0
            else (
                (time_electrostatics + time_total * (lambda_global - 1.0))
                / time_electrostatics
            )
        )
        lambda_sterics = (
            1.0
            if time_frame <= n_electrostatic_timesteps
            else (time_total / (time_total - time_electrostatics) * lambda_global)
        )

        return lambda_global, lambda_electrostatics, lambda_sterics

    def _simulate(
        self,
        coordinates: unit.Quantity,
        box_vectors: Optional[unit.Quantity],
        reverse_direction: bool,
    ) -> numpy.ndarray:
        """Evolve the state of the context according to a specific protocol."""

        _, lambda_electrostatics, lambda_sterics = self._compute_lambdas(
            0, reverse_direction
        )

        set_coordinates(self._context, coordinates, box_vectors)
        set_alchemical_lambdas(self._context, lambda_sterics, lambda_electrostatics)

        integrator: openmm.LangevinIntegrator = self._context.getIntegrator()

        frame_index = 0

        reduced_potentials = []

        stages = (
            (
                self._protocol.n_electrostatic_steps,
                self._protocol.n_steps_per_electrostatic_step,
            ),
            (
                self._protocol.n_steric_steps + 1,
                self._protocol.n_steps_per_steric_step,
            ),
        )

        for i, (n_lambda_steps, n_steps_per_lambda) in enumerate(
            stages if not reverse_direction else reversed(stages)
        ):

            for _ in tqdm(range(n_lambda_steps - int(i + 1 == len(stages)))):

                reduced_potential_old = self._compute_reduced_potential()

                (_, lambda_electrostatics, lambda_sterics) = self._compute_lambdas(
                    frame_index + 1, reverse_direction
                )

                set_alchemical_lambdas(
                    self._context, lambda_sterics, lambda_electrostatics
                )

                reduced_potential_new = self._compute_reduced_potential()

                reduced_potentials.append(
                    (reduced_potential_old, reduced_potential_new)
                )

                integrator.step(n_steps_per_lambda)
                frame_index += n_steps_per_lambda

        return numpy.array(reduced_potentials)

    def run(self) -> Tuple[float, float]:
        """Run the full simulation, restarting from where it left off if a previous
        attempt to run had already been made.
        """

        forward_potentials = self._simulate(*self._state_0, reverse_direction=False)
        reverse_potentials = self._simulate(*self._state_1, reverse_direction=True)

        forward_work = (forward_potentials[:, 1] - forward_potentials[:, 0]).sum()
        reverse_work = (reverse_potentials[:, 1] - reverse_potentials[:, 0]).sum()

        return forward_work, reverse_work
