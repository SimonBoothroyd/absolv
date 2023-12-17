import pytest
import pydantic

from absolv.config import EquilibriumProtocol, System


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

    def test_validate_solvent_a(self):
        with pytest.raises(
            pydantic.ValidationError, match="specified when `solvent_a` is not none"
        ):
            System(solutes={"C": 1}, solvent_a={}, solvent_b=None)

        system = System(solutes={"C": 1}, solvent_a={"O": 2}, solvent_b=None)
        assert system.solvent_a == {"O": 2}

    def test_validate_solvent_b(self):
        with pytest.raises(
            pydantic.ValidationError, match="specified when `solvent_b` is not none"
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
        with pytest.raises(
            pydantic.ValidationError, match="lambda lists must be the same"
        ):
            EquilibriumProtocol(
                lambda_sterics=lambda_sterics,
                lambda_electrostatics=lambda_electrostatics,
            )
