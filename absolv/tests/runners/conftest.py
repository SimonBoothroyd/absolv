import pytest

from absolv.models import (
    EquilibriumProtocol,
    NonEquilibriumProtocol,
    SimulationProtocol,
    State,
    SwitchingProtocol,
    System,
    TransferFreeEnergySchema,
)


@pytest.fixture()
def argon_eq_schema():

    return TransferFreeEnergySchema(
        system=System(solutes={"[#18]": 1}, solvent_a={"[#18]": 255}, solvent_b=None),
        state=State(temperature=85.5, pressure=1.0),
        alchemical_protocol_a=EquilibriumProtocol(
            minimization_protocol=None,
            equilibration_protocol=None,
            production_protocol=SimulationProtocol(
                n_iterations=1, n_steps_per_iteration=1
            ),
            lambda_sterics=[1.0, 0.5, 0.0],
            lambda_electrostatics=[0.0, 0.0, 0.0],
        ),
        alchemical_protocol_b=EquilibriumProtocol(
            minimization_protocol=None,
            equilibration_protocol=None,
            production_protocol=SimulationProtocol(
                n_iterations=1, n_steps_per_iteration=1
            ),
            lambda_sterics=[1.0, 0.0],
            lambda_electrostatics=[1.0, 1.0],
        ),
    )


@pytest.fixture()
def argon_neq_schema():

    return TransferFreeEnergySchema(
        system=System(solutes={"[#18]": 1}, solvent_a={"[#18]": 255}, solvent_b=None),
        state=State(temperature=85.5, pressure=1.0),
        alchemical_protocol_a=NonEquilibriumProtocol(
            switching_protocol=SwitchingProtocol(
                n_electrostatic_steps=0,
                n_steps_per_electrostatic_step=0,
                n_steric_steps=2,
                n_steps_per_steric_step=1,
            )
        ),
        alchemical_protocol_b=NonEquilibriumProtocol(
            switching_protocol=SwitchingProtocol(
                n_electrostatic_steps=0,
                n_steps_per_electrostatic_step=0,
                n_steric_steps=2,
                n_steps_per_steric_step=1,
            )
        ),
    )
