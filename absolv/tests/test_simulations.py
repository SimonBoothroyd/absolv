import copy
import os.path
import shutil
from typing import Tuple

import mdtraj
import numpy
import openmm
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.utilities import temporary_cd
from openmm import unit
from openmmtools import cache, mcmc, multistate

from absolv.factories.alchemical import OpenMMAlchemicalFactory
from absolv.factories.coordinate import PACKMOLCoordinateFactory
from absolv.models import (
    EquilibriumProtocol,
    MinimizationProtocol,
    SimulationProtocol,
    State,
    SwitchingProtocol,
)
from absolv.simulations import (
    AlchemicalOpenMMSimulation,
    EquilibriumOpenMMSimulation,
    NonEquilibriumOpenMMSimulation,
    RepexAlchemicalOpenMMSimulation,
    _BaseOpenMMSimulation,
    _OpenMMTopology,
)
from absolv.tests import BaseTemporaryDirTest, all_close, is_close
from absolv.utilities.openmm import (
    array_to_vectors,
    extract_coordinates,
    set_coordinates,
)


def _build_alchemical_lj_system(
    n_alchemical: int,
    n_persistent: int,
    epsilon: unit.Quantity = 125.7 * unit.kelvin * unit.MOLAR_GAS_CONSTANT_R,
    sigma: unit.Quantity = 3.345 * unit.angstrom,
) -> openmm.System:

    system = openmm.System()

    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(6.0 * unit.angstrom)
    force.setUseDispersionCorrection(False)

    system.addForce(force)

    for i in range(n_alchemical + n_persistent):

        system.addParticle(39.948)
        force.addParticle(0.0, sigma, epsilon)

    return OpenMMAlchemicalFactory.generate(
        system,
        [{i} for i in range(n_alchemical)],
        [{i} for i in range(n_alchemical, n_alchemical + n_persistent)],
    )


@pytest.fixture(scope="module")
def _alchemical_argon_system() -> Tuple[Topology, unit.Quantity, openmm.System]:

    topology, coordinates = PACKMOLCoordinateFactory.generate(
        [("[Ar]", 100)], target_density=1.0 * unit.grams / unit.milliliters
    )

    system = _build_alchemical_lj_system(1, 99)
    system.setDefaultPeriodicBoxVectors(*array_to_vectors(topology.box_vectors))

    return topology, coordinates, system


@pytest.fixture()
def alchemical_argon_system(
    _alchemical_argon_system,
) -> Tuple[Topology, unit.Quantity, openmm.System]:
    return copy.deepcopy(_alchemical_argon_system)


@pytest.fixture()
def alchemical_argon_eq_simulation(alchemical_argon_system):

    topology, coordinates, system = alchemical_argon_system

    protocol = EquilibriumProtocol(
        lambda_sterics=[1.0, 0.0],
        lambda_electrostatics=[1.0, 1.0],
        minimization_protocol=MinimizationProtocol(max_iterations=1),
        equilibration_protocol=SimulationProtocol(
            n_iterations=1, n_steps_per_iteration=1
        ),
        production_protocol=SimulationProtocol(n_iterations=1, n_steps_per_iteration=1),
        sampler="independent",
    )

    simulation = EquilibriumOpenMMSimulation(
        system,
        coordinates,
        topology.box_vectors,
        State(temperature=85.5, pressure=1.0),
        protocol,
        1,
        "CPU",
    )
    yield simulation
    del simulation._context


@pytest.fixture()
def repex_argon_eq_simulation(alchemical_argon_system):

    topology, coordinates, system = alchemical_argon_system

    protocol = EquilibriumProtocol(
        lambda_sterics=[1.0, 1.0, 0.0],
        lambda_electrostatics=[1.0, 0.0, 0.0],
        minimization_protocol=MinimizationProtocol(max_iterations=1),
        equilibration_protocol=SimulationProtocol(
            n_iterations=1, n_steps_per_iteration=1
        ),
        production_protocol=SimulationProtocol(n_iterations=2, n_steps_per_iteration=3),
        sampler="repex",
    )

    simulation = RepexAlchemicalOpenMMSimulation(
        system,
        coordinates,
        topology.box_vectors,
        State(temperature=85.5, pressure=1.0),
        protocol,
        "Reference",
    )
    yield simulation


@pytest.fixture()
def alchemical_argon_neq_simulation(alchemical_argon_system):

    topology, coordinates, system = alchemical_argon_system

    protocol = SwitchingProtocol(
        n_electrostatic_steps=1,
        n_steps_per_electrostatic_step=1,
        n_steric_steps=2,
        n_steps_per_steric_step=2,
        timestep=2.0 * unit.femtosecond,
        thermostat_friction=1.0 / unit.picosecond,
    )

    simulation = NonEquilibriumOpenMMSimulation(
        system,
        State(temperature=85.5, pressure=1.0),
        coordinates,
        topology.box_vectors,
        coordinates,
        topology.box_vectors,
        protocol,
        "Reference",
    )
    yield simulation
    del simulation._context


class TestBaseOpenMMSimulation(BaseTemporaryDirTest):
    def test_init(self, alchemical_argon_system):

        topology, coordinates, system = alchemical_argon_system

        simulation = _BaseOpenMMSimulation(
            system,
            coordinates,
            topology.box_vectors,
            State(temperature=85.5, pressure=0.5),
            "CPU",
        )

        assert isinstance(simulation._context, openmm.Context)

        context_coordinates, context_box_vectors = extract_coordinates(
            simulation._context
        )
        assert all_close(context_coordinates, coordinates)
        assert all_close(context_box_vectors, topology.box_vectors)

        assert isinstance(simulation._mock_topology, _OpenMMTopology)
        assert simulation._mock_topology.atoms() == list(range(100))

        expected_beta = 1.0 / (85.5 * unit.kelvin * unit.BOLTZMANN_CONSTANT_kB)
        assert is_close(expected_beta, simulation._beta)

        expected_pressure = 0.5 * unit.atmosphere
        assert is_close(expected_pressure, simulation._pressure)

    def test_current_state(self, alchemical_argon_eq_simulation):

        current_state = alchemical_argon_eq_simulation.current_state
        assert isinstance(current_state, openmm.State)

    def test_save_restore_state(self, alchemical_argon_eq_simulation):

        initial_coordinates, initial_box_vectors = extract_coordinates(
            alchemical_argon_eq_simulation._context
        )

        assert alchemical_argon_eq_simulation._restore_state("test") is False
        alchemical_argon_eq_simulation._save_state("test")
        assert os.path.isfile("test-state.xml")

        set_coordinates(
            alchemical_argon_eq_simulation._context,
            initial_coordinates + 1.0 * unit.angstrom,
            initial_box_vectors,
        )

        assert alchemical_argon_eq_simulation._restore_state("test") is True
        coordinates, box_vectors = extract_coordinates(
            alchemical_argon_eq_simulation._context
        )
        assert all_close(initial_coordinates, coordinates)
        assert all_close(initial_box_vectors, box_vectors)

    def test_compute_reduced_potential(self):

        topology = Topology.from_molecules([Molecule.from_smiles("[Ar]")] * 2)
        topology.box_vectors = (numpy.eye(3) * 12.0) * unit.angstrom

        system = _build_alchemical_lj_system(
            2, 0, 1.0 * unit.kilojoules_per_mole, 1.0 * unit.angstrom
        )

        coordinates = numpy.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * unit.angstrom

        simulation = _BaseOpenMMSimulation(
            system,
            coordinates,
            topology.box_vectors,
            State(temperature=3.0 * unit.kelvin, pressure=4.0 * unit.atmosphere),
            "CPU",
        )

        reduced_potential = simulation._compute_reduced_potential()

        expected_energy = 4.0 * (2.0**-12 - 2.0**-6) * unit.kilojoules_per_mole
        expected_volume = (12.0 * unit.angstrom) ** 3

        expected_reduced = expected_energy / (
            3.0 * unit.kelvin * unit.MOLAR_GAS_CONSTANT_R
        )
        expected_reduced += (
            (4.0 * unit.atmosphere)
            * expected_volume
            / (3.0 * unit.kelvin * unit.BOLTZMANN_CONSTANT_kB)
        )

        assert is_close(reduced_potential, expected_reduced)


class TestEquilibriumOpenMMSimulation(BaseTemporaryDirTest):
    def test_init(self, alchemical_argon_system):

        topology, coordinates, system = alchemical_argon_system

        protocol = EquilibriumProtocol(
            lambda_sterics=[1.0, 0.0],
            lambda_electrostatics=[1.0, 1.0],
            sampler="independent",
        )

        simulation = EquilibriumOpenMMSimulation(
            system,
            coordinates,
            topology.box_vectors,
            State(temperature=85.5, pressure=1.0),
            protocol,
            1,
            "CPU",
        )

        # Sanity check super was called
        assert isinstance(simulation._context, openmm.Context)

        assert simulation._protocol == protocol

        assert numpy.isclose(simulation._lambda_sterics, 0.0)
        assert numpy.isclose(simulation._lambda_electrostatics, 1.0)

        assert numpy.isclose(simulation._context.getParameter("lambda_sterics"), 0.0)
        assert numpy.isclose(
            simulation._context.getParameter("lambda_electrostatics"), 1.0
        )

    def test_minimize(self, alchemical_argon_eq_simulation):

        initial_coordinates, initial_box_vectors = extract_coordinates(
            alchemical_argon_eq_simulation._context
        )
        initial_energy = alchemical_argon_eq_simulation._context.getState(
            getEnergy=True
        ).getPotentialEnergy()

        assert not os.path.isfile("minimized-state.xml")
        alchemical_argon_eq_simulation._minimize()

        final_energy = alchemical_argon_eq_simulation._context.getState(
            getEnergy=True
        ).getPotentialEnergy()

        assert final_energy < initial_energy
        assert os.path.isfile("minimized-state.xml")

        set_coordinates(
            alchemical_argon_eq_simulation._context,
            initial_coordinates,
            initial_box_vectors,
        )

        alchemical_argon_eq_simulation._minimize()

        final_energy_2 = alchemical_argon_eq_simulation._context.getState(
            getEnergy=True
        ).getPotentialEnergy()

        # The minimization should have check-pointed so the energy should
        # be the same.
        assert is_close(final_energy, final_energy_2)

    def test_simulate(self, alchemical_argon_system, alchemical_argon_eq_simulation):

        topology, coordinates, _ = alchemical_argon_system
        topology.to_file("topology.pdb", coordinates)

        initial_energy = alchemical_argon_eq_simulation._context.getState(
            getEnergy=True
        ).getPotentialEnergy()

        assert not os.path.isfile("test-final-state.xml")
        assert not os.path.isfile("test-chk-state.xml")

        alchemical_argon_eq_simulation._simulate(
            alchemical_argon_eq_simulation._protocol.equilibration_protocol, "test"
        )

        final_energy_1 = alchemical_argon_eq_simulation._context.getState(
            getEnergy=True
        ).getPotentialEnergy()

        assert not is_close(final_energy_1, initial_energy)
        assert not os.path.isfile("test-chk-state.xml")  # Should have been cleaned-up
        assert os.path.isfile("test-trajectory.dcd")
        assert os.path.isfile("test-final-state.xml")

        trajectory_1 = mdtraj.load_dcd("test-trajectory.dcd", "topology.pdb")
        assert len(trajectory_1) == 1

        shutil.move("test-final-state.xml", "test-chk-state.xml")

        alchemical_argon_eq_simulation._simulate(
            SimulationProtocol(n_iterations=2, n_steps_per_iteration=1), "test"
        )

        final_energy_2 = alchemical_argon_eq_simulation._context.getState(
            getEnergy=True
        ).getPotentialEnergy()

        assert not is_close(final_energy_2, final_energy_1)

        trajectory_2 = mdtraj.load_dcd("test-trajectory.dcd", "topology.pdb")
        assert len(trajectory_2) == 2

        assert all_close(trajectory_1.xyz[0], trajectory_2.xyz[0])

        final_energy_3 = alchemical_argon_eq_simulation._context.getState(
            getEnergy=True
        ).getPotentialEnergy()

        assert is_close(final_energy_2, final_energy_3)

    def test_run(self, alchemical_argon_eq_simulation):

        alchemical_argon_eq_simulation.run(os.path.curdir)

        assert os.path.isfile("minimized-state.xml")
        assert os.path.isfile("equilibration-final-state.xml")
        assert os.path.isfile("production-final-state.xml")


class TestAlchemicalOpenMMSimulation(BaseTemporaryDirTest):
    def test_init(self, alchemical_argon_system):

        topology, coordinates, system = alchemical_argon_system

        protocol = EquilibriumProtocol(
            lambda_sterics=[1.0, 0.0],
            lambda_electrostatics=[1.0, 1.0],
            sampler="independent",
        )

        simulation = AlchemicalOpenMMSimulation(
            system,
            coordinates,
            topology.box_vectors,
            State(temperature=85.5, pressure=0.5),
            protocol,
            1,
            "CPU",
        )

        # Sanity check super was called
        assert isinstance(simulation._context, openmm.Context)

    def test_begin_end_iteration(self):

        topology = Topology.from_molecules([Molecule.from_smiles("[Ar]")] * 2)

        system = _build_alchemical_lj_system(
            1, 1, 1.0 * unit.kilojoules_per_mole, 1.0 * unit.angstrom
        )
        [*system.getForces()][0].setNonbondedMethod(
            openmm.NonbondedForce.CutoffNonPeriodic
        )

        coordinates = numpy.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * unit.angstrom

        simulation = AlchemicalOpenMMSimulation(
            system,
            coordinates,
            topology.box_vectors,
            State(temperature=3.0 * unit.kelvin, pressure=None),
            EquilibriumProtocol(
                lambda_sterics=[1.0, 0.0],
                lambda_electrostatics=[1.0, 1.0],
                sampler="independent",
            ),
            0,
            "CPU",
        )

        expected_reduced_potential = simulation._compute_reduced_potential()

        simulation._begin_iteration(0, "test")
        assert simulation._energies_file is None

        simulation._begin_iteration(0, "production")
        assert simulation._energies_file is not None

        simulation._end_iteration(0, "test")
        assert simulation._energies_file is not None

        simulation._end_iteration(0, "production")
        assert os.path.isfile("lambda-potentials.csv")

        lambda_potentials = numpy.genfromtxt("lambda-potentials.csv", delimiter=" ")
        assert lambda_potentials.shape == (2,)

        assert all_close(
            lambda_potentials, numpy.array([expected_reduced_potential, 0.0])
        )


class TestRepexAlchemicalOpenMMSimulation(BaseTemporaryDirTest):
    def test_init(self, alchemical_argon_system):
        topology, coordinates, system = alchemical_argon_system

        protocol = EquilibriumProtocol(
            lambda_sterics=[1.0, 0.0], lambda_electrostatics=[1.0, 1.0], sampler="repex"
        )

        simulation = RepexAlchemicalOpenMMSimulation(
            system,
            coordinates,
            topology.box_vectors,
            State(temperature=85.5, pressure=0.5),
            protocol,
            "CPU",
        )

        assert simulation._system == system

        assert simulation._coordinates.shape == coordinates.shape
        assert simulation._box_vectors.shape == topology.box_vectors.shape

        assert numpy.isclose(simulation._state.temperature, 85.5)
        assert numpy.isclose(simulation._state.pressure, 0.5)

        assert simulation._platform == "CPU"

        assert simulation._protocol == protocol

    def test_setup_sampler(
        self, alchemical_argon_system, repex_argon_eq_simulation, tmpdir
    ):

        topology, coordinates, _ = alchemical_argon_system

        expected_state_coordinates = [
            (coordinates + i * unit.angstrom, topology.box_vectors) for i in range(3)
        ]

        simulation = repex_argon_eq_simulation._setup_sampler(
            expected_state_coordinates, storage_path=os.path.join(tmpdir, "storage.nc")
        )

        assert cache.global_context_cache.platform.getName() == "Reference"

        assert simulation.n_states == 3
        assert simulation.n_replicas == 3

        assert (
            simulation.number_of_iterations
            == repex_argon_eq_simulation._protocol.production_protocol.n_iterations
        )

        for mcmc_move in simulation.mcmc_moves:
            assert isinstance(mcmc_move, mcmc.LangevinDynamicsMove)

            assert numpy.isclose(
                mcmc_move.timestep.value_in_unit(unit.femtosecond),
                repex_argon_eq_simulation._protocol.production_protocol.timestep,
            )
            assert numpy.isclose(
                mcmc_move.collision_rate.value_in_unit(unit.picosecond**-1),
                repex_argon_eq_simulation._protocol.production_protocol.thermostat_friction,
            )
            assert (
                mcmc_move.n_steps
                == repex_argon_eq_simulation._protocol.production_protocol.n_steps_per_iteration
            )

        for sampler_state, (expected_coordinates, expected_box_vectors) in zip(
            simulation.sampler_states, expected_state_coordinates
        ):

            assert numpy.allclose(
                expected_coordinates.value_in_unit(unit.angstrom),
                sampler_state.positions.value_in_unit(unit.angstrom),
            )
            assert numpy.allclose(
                expected_box_vectors.value_in_unit(unit.angstrom),
                sampler_state.box_vectors.value_in_unit(unit.angstrom),
            )

        for thermodynamic_state, expected_lambda_sterics, expected_lambda_elec in zip(
            simulation._thermodynamic_states,
            repex_argon_eq_simulation._protocol.lambda_sterics,
            repex_argon_eq_simulation._protocol.lambda_electrostatics,
        ):

            assert numpy.isclose(
                thermodynamic_state.temperature.value_in_unit(unit.kelvin), 85.5
            )

            assert len(thermodynamic_state._composable_states) == 1
            assert numpy.isclose(
                thermodynamic_state._composable_states[0].lambda_sterics,
                expected_lambda_sterics,
            )
            assert numpy.isclose(
                thermodynamic_state._composable_states[0].lambda_electrostatics,
                expected_lambda_elec,
            )

        assert os.path.isfile(os.path.join(tmpdir, "storage.nc"))

    def test_save_reduced_potentials(self, repex_argon_eq_simulation, tmpdir):

        with temporary_cd(str(tmpdir)):

            reporter = multistate.MultiStateReporter(
                "storage.nc", checkpoint_interval=1, open_mode="w"
            )

            reporter.write_energies(
                numpy.arange(9).reshape((3, 3)), numpy.eye(3), numpy.zeros((3, 0)), 0
            )
            reporter.write_replica_thermodynamic_states([0, 1, 2], 0)
            reporter.write_energies(
                numpy.arange(9).reshape((3, 3)), numpy.eye(3), numpy.zeros((3, 0)), 1
            )
            reporter.write_replica_thermodynamic_states([0, 1, 2], 1)

            reporter.write_energies(
                numpy.arange(9).reshape((3, 3)) + 9.0,
                numpy.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
                numpy.zeros((3, 0)),
                2,
            )
            reporter.write_replica_thermodynamic_states([2, 1, 0], 2)
            reporter.write_last_iteration(2)
            reporter.close()

            repex_argon_eq_simulation._save_reduced_potentials("storage.nc")

            saved_energies = numpy.stack(
                [
                    numpy.genfromtxt(
                        os.path.join(f"state-{state_index}", "lambda-potentials.csv")
                    )
                    for state_index in range(3)
                ]
            )
            assert saved_energies.shape == (3, 2, 3)

            # [iter][state_idx][state_e_idx]
            expected_energies = numpy.array(
                [
                    [[0.0, 1.0, 2.0], [15.0, 16.0, 17.0]],
                    [[3.0, 4.0, 5.0], [12.0, 13.0, 14.0]],
                    [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
                ]
            )

            assert numpy.allclose(saved_energies, expected_energies)

    def test_run(self, repex_argon_eq_simulation, tmpdir):

        with temporary_cd(str(tmpdir)):

            repex_argon_eq_simulation.run("")

            expected_files = [
                "minimized-state.xml",
                "equilibration-final-state.xml",
                "production-storage.nc",
                "production-storage_checkpoint.nc",
                os.path.join("state-0", "lambda-potentials.csv"),
                os.path.join("state-1", "lambda-potentials.csv"),
                os.path.join("state-2", "lambda-potentials.csv"),
            ]

            assert all(
                os.path.isfile(expected_file) for expected_file in expected_files
            )


class TestNonEquilibriumOpenMMSimulation(BaseTemporaryDirTest):
    def test_init(self, alchemical_argon_system):

        topology, coordinates, system = alchemical_argon_system

        protocol = SwitchingProtocol(
            n_electrostatic_steps=1,
            n_steps_per_electrostatic_step=10,
            n_steric_steps=2,
            n_steps_per_steric_step=20,
            timestep=0.5 * unit.femtosecond,
            thermostat_friction=2.0 / unit.picosecond,
        )

        simulation = NonEquilibriumOpenMMSimulation(
            system,
            State(temperature=85.5, pressure=1.0),
            coordinates,
            topology.box_vectors,
            -coordinates,
            topology.box_vectors * 2.0,
            protocol,
            "Reference",
        )

        # Sanity check super was called
        assert isinstance(simulation._context, openmm.Context)

        integrator = simulation._context.getIntegrator()
        assert is_close(integrator.getStepSize(), protocol.timestep * unit.femtoseconds)
        assert is_close(
            integrator.getFriction(), protocol.thermostat_friction / unit.picoseconds
        )

        assert simulation._protocol == protocol

        assert numpy.allclose(simulation._state_0[0], coordinates)
        assert numpy.allclose(simulation._state_1[0], -coordinates)

        assert numpy.allclose(simulation._state_0[1], topology.box_vectors)
        assert numpy.allclose(simulation._state_1[1], topology.box_vectors * 2.0)

    @pytest.mark.parametrize(
        "time_frame, reverse_direction, expected_lambda_global, "
        "expected_lambda_electrostatics, expected_lambda_sterics",
        [
            (0, False, 1.0, 1.0, 1.0),
            (10, False, 6.0 / 7.0, 2.0 / 3.0, 1.0),
            (20, False, 5.0 / 7.0, 1.0 / 3.0, 1.0),
            (30, False, 4.0 / 7.0, 0.0, 1.0),
            (50, False, 2.0 / 7.0, 0.0, 0.5),
            (70, False, 0.0 / 7.0, 0.0, 0.0),
            (0, True, 0.0, 0.0, 0.0),
            (20, True, 2.0 / 7.0, 0.0, 0.5),
            (40, True, 4.0 / 7.0, 0.0, 1.0),
            (50, True, 5.0 / 7.0, 1.0 / 3.0, 1.0),
            (60, True, 6.0 / 7.0, 2.0 / 3.0, 1.0),
            (70, True, 7.0 / 7.0, 1.0, 1.0),
        ],
    )
    def test_compute_lambdas(
        self,
        alchemical_argon_eq_simulation,
        time_frame,
        reverse_direction,
        expected_lambda_global,
        expected_lambda_electrostatics,
        expected_lambda_sterics,
    ):

        simulation = NonEquilibriumOpenMMSimulation.__new__(
            NonEquilibriumOpenMMSimulation
        )
        simulation._protocol = SwitchingProtocol(
            n_electrostatic_steps=3,
            n_steps_per_electrostatic_step=10,
            n_steric_steps=2,
            n_steps_per_steric_step=20,
            timestep=0.5 * unit.femtosecond,
            thermostat_friction=2.0 / unit.picosecond,
        )

        (
            actual_lambda_global,
            actual_lambda_electrostatics,
            actual_lambda_sterics,
        ) = simulation._compute_lambdas(time_frame, reverse_direction)

        print(expected_lambda_global, actual_lambda_global)

        assert numpy.isclose(expected_lambda_global, actual_lambda_global)
        assert numpy.isclose(
            expected_lambda_electrostatics, actual_lambda_electrostatics
        )
        assert numpy.isclose(expected_lambda_sterics, actual_lambda_sterics)

    @pytest.mark.parametrize(
        "reverse_direction, expected_frame_indices",
        [
            (False, [(0, 10), (10, 10), (20, 10), (30, 20), (50, 20)]),
            (True, [(0, 20), (20, 20), (40, 10), (50, 10), (60, 10)]),
        ],
    )
    def test_enumerate_frames(
        self, alchemical_argon_neq_simulation, reverse_direction, expected_frame_indices
    ):

        simulation = alchemical_argon_neq_simulation

        simulation._protocol = SwitchingProtocol(
            n_electrostatic_steps=3,
            n_steps_per_electrostatic_step=10,
            n_steric_steps=2,
            n_steps_per_steric_step=20,
        )

        frame_indices = [*simulation._enumerate_frames(reverse_direction)]
        assert frame_indices == expected_frame_indices

    @pytest.mark.parametrize(
        "reverse_direction, expected_lambda_values",
        [
            (
                False,
                [
                    (1.0, 1.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (0.5, 0.0),
                    (0.5, 0.0),
                    (0.0, 0.0),
                ],
            ),
            (
                True,
                [
                    (0.0, 0.0),
                    (0.5, 0.0),
                    (0.5, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 1.0),
                ],
            ),
        ],
    )
    def test_simulate(
        self,
        alchemical_argon_system,
        alchemical_argon_neq_simulation,
        monkeypatch,
        reverse_direction,
        expected_lambda_values,
    ):

        topology, coordinates, system = alchemical_argon_system

        lambda_values = []

        original_compute_reduced_potential = (
            alchemical_argon_neq_simulation._compute_reduced_potential
        )

        def mock_compute_reduced_potential():

            lambda_values.append(
                tuple(
                    alchemical_argon_neq_simulation._context.getParameter(lambda_name)
                    for lambda_name in ("lambda_sterics", "lambda_electrostatics")
                )
            )

            return original_compute_reduced_potential()

        monkeypatch.setattr(
            alchemical_argon_neq_simulation,
            "_compute_reduced_potential",
            mock_compute_reduced_potential,
        )

        reduced_potentials = alchemical_argon_neq_simulation._simulate(
            coordinates, topology.box_vectors, reverse_direction
        )
        assert reduced_potentials.shape == (3, 2)
        assert lambda_values == expected_lambda_values

    def test_run(self, alchemical_argon_neq_simulation):

        forward_work, reverse_work = alchemical_argon_neq_simulation.run()

        assert not is_close(forward_work, 0.0)
        assert not is_close(reverse_work, 0.0)
