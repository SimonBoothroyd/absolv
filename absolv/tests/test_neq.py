import numpy
import openff.toolkit
import openmm
import openmm.unit
import pytest

import absolv.config
import absolv.fep
import absolv.setup
from absolv.neq import _compute_lambdas, _enumerate_frames, _simulate, run_neq


@pytest.fixture
def mock_simulation():
    topology, coords = absolv.setup.setup_system([("C", 2)])
    topology.box_vectors *= 100.0

    expected_temperature = 298.15 * openmm.unit.kelvin
    expected_pressure = 1.0 * openmm.unit.atmosphere

    force_field = openff.toolkit.ForceField("openff-2.1.0.offxml")
    system = force_field.create_openmm_system(topology)
    system.addForce(openmm.MonteCarloBarostat(expected_pressure, expected_temperature))
    system = absolv.fep.apply_fep(system, [{0, 1, 2, 3, 4}], [{5, 6, 7, 8, 9}])

    simulation = openmm.app.Simulation(
        topology,
        system,
        openmm.LangevinIntegrator(
            expected_temperature,
            1.0 / openmm.unit.picosecond,
            2.0 * openmm.unit.femtosecond,
        ),
    )
    simulation.context.setPositions(coords)
    return simulation


@pytest.mark.parametrize(
    "step, reverse_direction, expected_lambda_global, "
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
    step,
    reverse_direction,
    expected_lambda_global,
    expected_lambda_electrostatics,
    expected_lambda_sterics,
):
    protocol = absolv.config.SwitchingProtocol(
        n_electrostatic_steps=3,
        n_steps_per_electrostatic_step=10,
        n_steric_steps=2,
        n_steps_per_steric_step=20,
    )

    (
        actual_lambda_global,
        actual_lambda_electrostatics,
        actual_lambda_sterics,
    ) = _compute_lambdas(
        protocol, step, 0.5 * openmm.unit.femtosecond, reverse_direction
    )

    assert numpy.isclose(expected_lambda_global, actual_lambda_global)
    assert numpy.isclose(expected_lambda_electrostatics, actual_lambda_electrostatics)
    assert numpy.isclose(expected_lambda_sterics, actual_lambda_sterics)


@pytest.mark.parametrize(
    "reverse_direction, expected_frame_indices",
    [
        (False, [(0, 10), (10, 10), (20, 10), (30, 20), (50, 20)]),
        (True, [(0, 20), (20, 20), (40, 10), (50, 10), (60, 10)]),
    ],
)
def test_enumerate_frames(reverse_direction, expected_frame_indices):
    protocol = absolv.config.SwitchingProtocol(
        n_electrostatic_steps=3,
        n_steps_per_electrostatic_step=10,
        n_steric_steps=2,
        n_steps_per_steric_step=20,
    )

    frame_indices = [*_enumerate_frames(protocol, reverse_direction)]
    assert frame_indices == expected_frame_indices


@pytest.mark.parametrize(
    "reverse_direction, expected_lambda_values, expected_steps",
    [
        (False, [(1.0, 1.0), (1.0, 0.0), (0.5, 0.0), (0.0, 0.0)], [1, 2, 2]),
        (True, [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0)], [2, 2, 1]),
    ],
)
def test_simulate(
    reverse_direction, expected_lambda_values, expected_steps, mocker, mock_simulation
):
    state = mock_simulation.context.getState(getPositions=True)

    spied_set_lambda = mocker.spy(absolv.fep, "set_fep_lambdas")
    mock_step = mocker.patch.object(mock_simulation, "step")

    protocol = absolv.config.SwitchingProtocol(
        n_electrostatic_steps=1,
        n_steps_per_electrostatic_step=1,
        n_steric_steps=2,
        n_steps_per_steric_step=2,
    )

    reduced_potentials = _simulate(mock_simulation, state, protocol, reverse_direction)

    lambda_values = [call.args[1:] for call in spied_set_lambda.call_args_list]
    assert lambda_values == expected_lambda_values

    mock_step.assert_has_calls([mocker.call(step) for step in expected_steps])
    assert reduced_potentials.shape == (3, 2)


def test_run(mock_simulation, mocker):
    state = mock_simulation.context.getState(getPositions=True)

    protocol = absolv.config.SwitchingProtocol(
        n_electrostatic_steps=1,
        n_steps_per_electrostatic_step=1,
        n_steric_steps=2,
        n_steps_per_steric_step=2,
    )

    mock_simulate = mocker.patch(
        "absolv.neq._simulate",
        autospec=True,
        side_effect=[
            numpy.array([[1.0, 3.0], [4.0, 9.0]]),
            numpy.array([[2.0, 1.0], [3.0, 0.0]]),
        ],
    )

    forward_work, reverse_work = run_neq(mock_simulate, state, state, protocol)

    assert numpy.isclose(forward_work, 7.0)
    assert numpy.isclose(reverse_work, -4.0)
