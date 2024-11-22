import femto.md.constants
import numpy
import openff.toolkit
import openff.units
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


def test_rebuild_topology():
    ff = openff.toolkit.ForceField("tip4p_fb.offxml", "openff-2.0.0.offxml")

    v_site_handler = ff.get_parameter_handler("VirtualSites")
    v_site_handler.add_parameter(
        {
            "type": "DivalentLonePair",
            "match": "once",
            "smirks": "[*:2][#7:1][*:3]",
            "distance": 0.4 * openff.units.unit.angstrom,
            "epsilon": 0.0 * openff.units.unit.kilojoule_per_mole,
            "sigma": 0.1 * openff.units.unit.nanometer,
            "outOfPlaneAngle": 0.0 * openff.units.unit.degree,
            "charge_increment1": 0.0 * openff.units.unit.elementary_charge,
            "charge_increment2": 0.0 * openff.units.unit.elementary_charge,
            "charge_increment3": 0.0 * openff.units.unit.elementary_charge,
        }
    )

    solute = openff.toolkit.Molecule.from_smiles("c1ccncc1")
    solute.generate_conformers(n_conformers=1)
    solvent = openff.toolkit.Molecule.from_smiles("O")
    solvent.generate_conformers(n_conformers=1)

    orig_coords = (
        numpy.vstack(
            [
                solute.conformers[0].m_as("angstrom"),
                solvent.conformers[0].m_as("angstrom") + numpy.array([10.0, 0.0, 0.0]),
                solvent.conformers[0].m_as("angstrom") + numpy.array([20.0, 0.0, 0.0]),
            ]
        )
        * openmm.unit.angstrom
    )

    expected_box_vectors = numpy.eye(3) * 30.0

    orig_top = openff.toolkit.topology.Topology.from_molecules(
        [solute, solvent, solvent]
    )
    orig_top.box_vectors = expected_box_vectors * openmm.unit.angstrom

    system = ff.create_openmm_system(orig_top)

    n_v_sites = sum(
        1 for i in range(system.getNumParticles()) if system.isVirtualSite(i)
    )
    assert n_v_sites == 3

    top, coords = absolv.runner._rebuild_topology(orig_top, orig_coords, system)

    found_atoms = [
        (
            atom.name,
            atom.element.symbol if atom.element is not None else None,
            atom.residue.index,
            atom.residue.name,
        )
        for atom in top.atoms()
    ]
    expected_atoms = [
        ("C1x", "C", 0, "UNK"),
        ("C2x", "C", 0, "UNK"),
        ("C3x", "C", 0, "UNK"),
        ("N1x", "N", 0, "UNK"),
        ("C4x", "C", 0, "UNK"),
        ("C5x", "C", 0, "UNK"),
        ("H1x", "H", 0, "UNK"),
        ("H2x", "H", 0, "UNK"),
        ("H3x", "H", 0, "UNK"),
        ("H4x", "H", 0, "UNK"),
        ("H5x", "H", 0, "UNK"),
        ("OW", "O", 1, "HOH"),
        ("HW1", "H", 1, "HOH"),
        ("HW2", "H", 1, "HOH"),
        ("OW", "O", 2, "HOH"),
        ("HW1", "H", 2, "HOH"),
        ("HW2", "H", 2, "HOH"),
        ("X1x", None, 3, "UNK"),
        ("X1x", None, 4, "UNK"),
        ("X1x", None, 5, "UNK"),
    ]

    assert found_atoms == expected_atoms

    expected_coords = numpy.array(
        [
            [0.00241, 0.10097, -0.05663],
            [-0.11673, 0.03377, -0.03801],
            [-0.11801, -0.08778, 0.03043],
            [-0.00041, -0.1375, 0.07759],
            [0.112, -0.068, 0.05657],
            [0.12301, 0.0524, -0.00964],
            [4e-05, 0.19505, -0.11016],
            [-0.2116, 0.07095, -0.0744],
            [-0.20828, -0.14529, 0.04827],
            [0.19995, -0.11721, 0.09863],
            [0.21761, 0.10263, -0.02266],
            [0.99992, 0.03664, 0.0],
            [0.91877, -0.01835, 0.0],
            [1.08131, -0.01829, 0.0],
            [1.99992, 0.03664, 0.0],
            [1.91877, -0.01835, 0.0],
            [2.08131, -0.01829, 0.0],
            [0.0011, -0.1722, 0.09743],
            [0.99994, 0.02611, 0.0],
            [1.99994, 0.02611, 0.0],
        ]
    )  # manually visually inspected

    assert coords.shape == expected_coords.shape
    assert numpy.allclose(coords, expected_coords, atol=1.0e-5)

    box_vectors = top.getPeriodicBoxVectors().value_in_unit(openmm.unit.angstrom)
    box_vectors = numpy.array(box_vectors)

    assert numpy.allclose(box_vectors, expected_box_vectors)

    expected_bonds = [
        ("C1x", "C2x"),
        ("C2x", "C3x"),
        ("C3x", "N1x"),
        ("N1x", "C4x"),
        ("C4x", "C5x"),
        ("C5x", "C1x"),
        ("C1x", "H1x"),
        ("C2x", "H2x"),
        ("C3x", "H3x"),
        ("C4x", "H4x"),
        ("C5x", "H5x"),
        ("C1x", "C2x"),
        ("C1x", "C3x"),
        ("C1x", "C2x"),
        ("C1x", "C3x"),
    ]

    actual_bonds = [(bond.atom1.name, bond.atom2.name) for bond in top.bonds()]
    assert actual_bonds == expected_bonds


def test_setup_fn():
    system = absolv.config.System(
        solutes={"[Na+]": 1, "[Cl-]": 1}, solvent_a=None, solvent_b={"O": 1}
    )

    prepared_system_a, prepared_system_b = absolv.runner.setup(
        system, MOCK_CONFIG_EQ, openff.toolkit.ForceField("openff-2.0.0.offxml")
    )

    assert prepared_system_a.system.getNumParticles() == 2
    assert prepared_system_b.system.getNumParticles() == 5

    assert prepared_system_a.topology.getPeriodicBoxVectors() is None
    assert prepared_system_b.topology.getPeriodicBoxVectors() is not None


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
