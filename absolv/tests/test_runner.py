import femto.md.constants
import openff.toolkit
import openmm.unit
import pytest

import absolv.config
import absolv.runner

DEFAULT_TEMPERATURE = 298.15 * openmm.unit.kelvin
DEFAULT_PRESSURE = 1.0 * openmm.unit.atmosphere


MOCK_CONFIG_NEQ = absolv.config.Config(
    temperature=DEFAULT_TEMPERATURE,
    pressure=DEFAULT_PRESSURE,
    alchemical_protocol_a=absolv.config.NonEquilibriumProtocol(
        equilibration_protocol=absolv.config.SimulationProtocol(n_steps=1),
        production_protocol=absolv.config.SimulationProtocol(n_steps=2),
        production_report_interval=1,
        switching_protocol=absolv.config.SwitchingProtocol(
            n_electrostatic_steps=1,
            n_steps_per_electrostatic_step=1,
            n_steric_steps=0,
            n_steps_per_steric_step=0,
        ),
    ),
    alchemical_protocol_b=absolv.config.NonEquilibriumProtocol(
        equilibration_protocol=absolv.config.SimulationProtocol(n_steps=1),
        production_protocol=absolv.config.SimulationProtocol(n_steps=2),
        production_report_interval=1,
        switching_protocol=absolv.config.SwitchingProtocol(
            n_electrostatic_steps=1,
            n_steps_per_electrostatic_step=1,
            n_steric_steps=1,
            n_steps_per_steric_step=1,
        ),
    ),
)
MOCK_CONFIG_EQ = absolv.config.Config(
    temperature=DEFAULT_TEMPERATURE,
    pressure=DEFAULT_PRESSURE,
    alchemical_protocol_a=absolv.config.EquilibriumProtocol(
        equilibration_protocol=absolv.config.SimulationProtocol(n_steps=1),
        production_protocol=absolv.config.HREMDProtocol(
            n_steps_per_cycle=1, n_cycles=1, n_warmup_steps=0
        ),
        lambda_sterics=[1.0],
        lambda_electrostatics=[1.0],
    ),
    alchemical_protocol_b=absolv.config.EquilibriumProtocol(
        equilibration_protocol=absolv.config.SimulationProtocol(n_steps=1),
        production_protocol=absolv.config.HREMDProtocol(
            n_steps_per_cycle=1, n_cycles=1, n_warmup_steps=0
        ),
        lambda_sterics=[1.0],
        lambda_electrostatics=[1.0],
    ),
)


def test_setup_fn():
    system = absolv.config.System(
        solutes={"[Na+]": 1, "[Cl-]": 1}, solvent_a=None, solvent_b={"O": 1}
    )

    prepared_system_a, prepared_system_b = absolv.runner.setup(
        system, MOCK_CONFIG_EQ, openff.toolkit.ForceField("openff-2.0.0.offxml")
    )

    assert prepared_system_a.system.getNumParticles() == 2
    assert prepared_system_b.system.getNumParticles() == 5

    assert prepared_system_a.topology.box_vectors is None
    assert prepared_system_b.topology.box_vectors is not None


@pytest.mark.parametrize(
    "run_fn, config",
    [(absolv.runner.run_neq, MOCK_CONFIG_NEQ), (absolv.runner.run_eq, MOCK_CONFIG_EQ)],
)
def test_run(run_fn, config):
    system = absolv.config.System(
        solutes={"[Na+]": 1, "[Cl-]": 1}, solvent_a=None, solvent_b=None
    )

    prepared_system_a, prepared_system_b = absolv.runner.setup(
        system, config, openff.toolkit.ForceField("openff-2.0.0.offxml")
    )
    result = run_fn(
        config,
        prepared_system_a,
        prepared_system_b,
        femto.md.constants.OpenMMPlatform.REFERENCE,
    )
    assert isinstance(result, absolv.config.Result)
