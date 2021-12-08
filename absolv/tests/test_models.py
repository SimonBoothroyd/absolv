import numpy
import pytest
from openmm import unit
from pydantic import ValidationError

from absolv.models import (
    DeltaG,
    EquilibriumProtocol,
    MinimizationProtocol,
    SimulationProtocol,
    State,
    SwitchingProtocol,
    System,
    TransferFreeEnergyResult,
)
from absolv.tests import is_close


class TestSystem:
    def test_n_solute_molecules(self):

        system = System(solutes={"CO": 2, "CCO": 3}, solvent_a={"O": 1}, solvent_b=None)
        assert system.n_solute_molecules == 5

    @pytest.mark.parametrize("solvent_a, n_expected", [({"O": 3}, 3), (None, 0)])
    def test_n_solvent_molecules_a(self, solvent_a, n_expected):
        system = System(
            solutes={
                "CO": 1,
            },
            solvent_a=solvent_a,
            solvent_b={"O": 5},
        )
        assert system.n_solvent_molecules_a == n_expected

    @pytest.mark.parametrize("solvent_b, n_expected", [({"O": 5}, 5), (None, 0)])
    def test_n_solvent_molecules_b(self, solvent_b, n_expected):
        system = System(
            solutes={
                "CO": 1,
            },
            solvent_a={"O": 3},
            solvent_b=solvent_b,
        )
        assert system.n_solvent_molecules_b == n_expected

    def test_validate_solutes(self):

        with pytest.raises(
            ValidationError, match="at least one solute must be specified"
        ):
            System(solutes={}, solvent_a=None, solvent_b=None)

        system = System(solutes={"C": 1}, solvent_a=None, solvent_b=None)
        assert system.solutes == {"C": 1}

    def test_validate_solvent_a(self):

        with pytest.raises(
            ValidationError, match="specified when `solvent_a` is not none"
        ):
            System(solutes={"C": 1}, solvent_a={}, solvent_b=None)

        system = System(solutes={"C": 1}, solvent_a={"O": 2}, solvent_b=None)
        assert system.solvent_a == {"O": 2}

    def test_validate_solvent_b(self):

        with pytest.raises(
            ValidationError, match="specified when `solvent_b` is not none"
        ):
            System(solutes={"C": 1}, solvent_a=None, solvent_b={})

        system = System(solutes={"C": 1}, solvent_a=None, solvent_b={"O": 2})
        assert system.solvent_b == {"O": 2}

    def test_to_components(self):

        system = System(
            solutes={"CO": 1, "CCO": 2}, solvent_a={"O": 3}, solvent_b={"OCO": 4}
        )

        components_a, components_b = system.to_components()

        assert components_a == [("CO", 1), ("CCO", 2), ("O", 3)]
        assert components_b == [("CO", 1), ("CCO", 2), ("OCO", 4)]


class TestState:
    def test_unit_validation(self):

        state = State(
            temperature=298.0 * unit.kelvin, pressure=101.325 * unit.kilopascals
        )

        assert is_close(state.temperature, 298.0)
        assert is_close(state.pressure, 1.0)


class TestMinimizationProtocol:
    def test_unit_validation(self):

        protocol = MinimizationProtocol(
            tolerance=1.0 * unit.kilojoule_per_mole / unit.angstrom
        )

        assert is_close(protocol.tolerance, 10.0)


class TestSimulationProtocol:
    def test_unit_validation(self):

        protocol = SimulationProtocol(
            n_steps_per_iteration=1,
            n_iterations=1,
            timestep=0.002 * unit.picoseconds,
            thermostat_friction=0.003 / unit.femtoseconds,
        )

        assert is_close(protocol.timestep, 2.0)
        assert is_close(protocol.thermostat_friction, 3.0)


class TestEquilibriumProtocol:
    def test_n_states(self):

        protocol = EquilibriumProtocol(
            lambda_sterics=[1.0, 0.5, 0.0], lambda_electrostatics=[1.0, 1.0, 1.0]
        )
        assert protocol.n_states == 3

    @pytest.mark.parametrize(
        "lambda_sterics, lambda_electrostatics",
        [([1.0, 0.5, 0.0], [1.0, 1.0]), ([1.0, 0.5], [1.0, 1.0, 1.0])],
    )
    def test_validate_lambda_lengths(self, lambda_sterics, lambda_electrostatics):

        with pytest.raises(ValidationError, match="lambda lists must be the same"):

            EquilibriumProtocol(
                lambda_sterics=lambda_sterics,
                lambda_electrostatics=lambda_electrostatics,
            )


class TestSwitchingProtocol:
    def test_unit_validation(self):

        protocol = SwitchingProtocol(
            n_electrostatic_steps=6250,
            n_steps_per_electrostatic_step=1,
            n_steric_steps=18750,
            n_steps_per_steric_step=1,
            timestep=0.002 * unit.picoseconds,
            thermostat_friction=0.003 / unit.femtoseconds,
        )

        assert is_close(protocol.timestep, 2.0)
        assert is_close(protocol.thermostat_friction, 3.0)


class TestDeltaG:
    def test_add(self):

        value_a = DeltaG(value=1.0, std_error=2.0)
        value_b = DeltaG(value=3.0, std_error=4.0)

        result = value_a + value_b

        assert is_close(result.value, 4.0)
        assert is_close(result.std_error, numpy.sqrt(20))

    def test_sub(self):

        value_a = DeltaG(value=1.0, std_error=2.0)
        value_b = DeltaG(value=3.0, std_error=4.0)

        result = value_b - value_a

        assert is_close(result.value, 2.0)
        assert is_close(result.std_error, numpy.sqrt(20))


class TestTransferFreeEnergyResult:
    @pytest.fixture()
    def free_energy_result(self, argon_eq_schema):

        return TransferFreeEnergyResult(
            input_schema=argon_eq_schema,
            delta_g_solvent_a=DeltaG(value=1.0, std_error=2.0),
            delta_g_solvent_b=DeltaG(value=3.0, std_error=4.0),
        )

    def test_delta_g_from_a_to_b(self, free_energy_result):

        delta_g = free_energy_result.delta_g_from_a_to_b

        assert is_close(delta_g.value, -2.0)
        assert is_close(delta_g.std_error, numpy.sqrt(20))

    def test_delta_g_from_b_to_a(self, free_energy_result):

        delta_g = free_energy_result.delta_g_from_b_to_a

        assert is_close(delta_g.value, 2.0)
        assert is_close(delta_g.std_error, numpy.sqrt(20))

    def test_boltzmann_temperature(self, free_energy_result):

        value = free_energy_result._boltzmann_temperature
        assert is_close(value, 85.5 * unit.kelvin * unit.MOLAR_GAS_CONSTANT_R)

    def test_delta_g_from_a_to_b_with_units(self, free_energy_result):

        value, std_error = free_energy_result.delta_g_from_a_to_b_with_units

        assert is_close(value, -2.0 * 85.5 * unit.kelvin * unit.MOLAR_GAS_CONSTANT_R)
        assert is_close(
            std_error, numpy.sqrt(20) * 85.5 * unit.kelvin * unit.MOLAR_GAS_CONSTANT_R
        )

    def test_delta_g_from_b_to_a_with_units(self, free_energy_result):

        value, std_error = free_energy_result.delta_g_from_b_to_a_with_units

        assert is_close(value, 2.0 * 85.5 * unit.kelvin * unit.MOLAR_GAS_CONSTANT_R)
        assert is_close(
            std_error, numpy.sqrt(20) * 85.5 * unit.kelvin * unit.MOLAR_GAS_CONSTANT_R
        )

    def test_str(self, free_energy_result):
        assert (
            str(free_energy_result)
            == "ΔG a->b=-0.340 kcal/mol ΔG a->b std=0.760 kcal/mol"
        )

    def test_repr(self, free_energy_result):
        assert repr(free_energy_result) == (
            "TransferFreeEnergyResult(ΔG a->b=-0.340 kcal/mol ΔG a->b std=0.760 kcal/mol)"
        )
