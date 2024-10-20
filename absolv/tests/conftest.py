import pathlib

import numpy
import openff.toolkit
import openff.units
import openff.units.openmm
import openmm
import openmm.unit
import pytest

from absolv.config import (
    Config,
    EquilibriumProtocol,
    NonEquilibriumProtocol,
    SimulationProtocol,
    SwitchingProtocol,
    System,
)

MOLAR_GAS_CONSTANT_R = openff.units.openmm.from_openmm(openmm.unit.MOLAR_GAS_CONSTANT_R)


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture()
def test_data_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def tip4p_nacl_lj_force_field() -> openff.toolkit.ForceField:
    force_field = openff.toolkit.ForceField()

    constraint_handler = force_field.get_parameter_handler("Constraints")
    constraint_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0:2]-[#1]",
            "distance": 0.9572 * openff.units.unit.angstrom,
        }
    )
    constraint_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0]-[#1:2]",
            "distance": 1.5139 * openff.units.unit.angstrom,
        }
    )

    vdw_handler = force_field.get_parameter_handler("vdW")
    vdw_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
            "epsilon": 0.0 * openff.units.unit.kilojoule_per_mole,
            "sigma": 1.0 * openff.units.unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
            "epsilon": 78.0 * openff.units.unit.kelvin * MOLAR_GAS_CONSTANT_R,
            "sigma": 3.154 * openff.units.unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#11+1:1]",
            "epsilon": 0.0874393 * openff.units.unit.kilocalories_per_mole,
            "rmin_half": 1.369 * openff.units.unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#17X0-1:1]",
            "epsilon": 0.035591 * openff.units.unit.kilocalories_per_mole,
            "rmin_half": 2.513 * openff.units.unit.angstrom,
        }
    )

    force_field.get_parameter_handler("Electrostatics")
    force_field.get_parameter_handler(
        "ChargeIncrementModel",
        {"version": "0.3", "partial_charge_method": "formal_charge"},
    )

    virtual_site_handler = force_field.get_parameter_handler("VirtualSites")
    virtual_site_handler.add_parameter(
        {
            "smirks": "[#1:2]-[#8X2H2+0:1]-[#1:3]",
            "type": "DivalentLonePair",
            "distance": -0.106 * openff.units.unit.angstrom,
            "outOfPlaneAngle": 0.0 * openff.units.unit.degrees,
            "match": "once",
            "charge_increment2": 1.0552 * 0.5 * openff.units.unit.elementary_charge,
            "charge_increment1": 0.0 * openff.units.unit.elementary_charge,
            "charge_increment3": 1.0552 * 0.5 * openff.units.unit.elementary_charge,
        }
    )

    return force_field


@pytest.fixture(scope="module")
def tip4p_meoh_de_force_field() -> openff.toolkit.ForceField:
    force_field = openff.toolkit.ForceField(load_plugins=True)

    constraint_handler = force_field.get_parameter_handler("Constraints")
    constraint_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0:2]-[#1]",
            "distance": 0.9572 * openff.units.unit.angstrom,
        }
    )
    constraint_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0]-[#1:2]",
            "distance": 1.5139 * openff.units.unit.angstrom,
        }
    )

    vdw_handler = force_field.get_parameter_handler(
        "DoubleExponential",
        handler_kwargs={
            "version": "0.3",
            "scale14": 0.5,
            "alpha": 16.789,
            "beta": 4.592,
        },
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#6:1]",
            "epsilon": 0.109 * openff.units.unit.kilocalories_per_mole,
            "r_min": 3.793 * openff.units.unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#1:1]",
            "epsilon": 0.0158 * openff.units.unit.kilocalories_per_mole,
            "r_min": 2.968 * openff.units.unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#8:1]",
            "epsilon": 0.209 * openff.units.unit.kilocalories_per_mole,
            "r_min": 3.364 * openff.units.unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
            "epsilon": 0.0 * openff.units.unit.kilojoule_per_mole,
            "r_min": 1.0 * openff.units.unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
            "epsilon": 78.0 * openff.units.unit.kelvin * MOLAR_GAS_CONSTANT_R,
            "r_min": 3.154 * openff.units.unit.angstrom,
        }
    )

    force_field.get_parameter_handler("Electrostatics")

    charge_handler = force_field.get_parameter_handler(
        "LibraryCharges",
        {"version": "0.3"},
    )
    charge_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0:2]-[#1:3]",
            "charge": [0.0, 0.0, 0.0] * openff.units.unit.elementary_charge,
        }
    )
    charge_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#6X4:2]-[#8:3]-[#1:4]",
            "charge": [0.0287, 0.1167, -0.5988, 0.396]
            * openff.units.unit.elementary_charge,
        }
    )

    virtual_site_handler = force_field.get_parameter_handler(
        "DoubleExponentialVirtualSites"
    )
    virtual_site_handler.add_parameter(
        {
            "smirks": "[#1:2]-[#8X2H2+0:1]-[#1:3]",
            "type": "DivalentLonePair",
            "distance": -0.106 * openff.units.unit.angstrom,
            "outOfPlaneAngle": 0.0 * openff.units.unit.degrees,
            "match": "once",
            "charge_increment2": 1.0552 * 0.5 * openff.units.unit.elementary_charge,
            "charge_increment1": 0.0 * openff.units.unit.elementary_charge,
            "charge_increment3": 1.0552 * 0.5 * openff.units.unit.elementary_charge,
            "epsilon": 0.0 * openff.units.unit.kilojoule / openff.units.unit.mole,
            "r_min": 1.0 * openff.units.unit.angstrom,
        }
    )

    return force_field


@pytest.fixture(scope="module")
def aq_nacl_topology() -> openff.toolkit.Topology:
    topology = openff.toolkit.Topology.from_molecules(
        [
            openff.toolkit.Molecule.from_smiles("[Na+]"),
            openff.toolkit.Molecule.from_smiles("[Cl-]"),
        ]
        + [openff.toolkit.Molecule.from_smiles("O")] * 2
    )
    topology.box_vectors = (numpy.eye(3) * 5.0) * openff.units.unit.nanometers

    return topology


@pytest.fixture(scope="module")
def aq_meoh_topology() -> openff.toolkit.Topology:
    topology = openff.toolkit.Topology.from_molecules(
        [openff.toolkit.Molecule.from_smiles("CO")]
        + [openff.toolkit.Molecule.from_smiles("O")] * 2
    )
    topology.box_vectors = (numpy.eye(3) * 5.0) * openff.units.unit.nanometers

    return topology


@pytest.fixture(scope="module")
def aq_nacl_lj_system(tip4p_nacl_lj_force_field, aq_nacl_topology) -> openmm.System:
    return tip4p_nacl_lj_force_field.create_openmm_system(aq_nacl_topology)


@pytest.fixture(scope="module")
def aq_meoh_de_system(tip4p_meoh_de_force_field, aq_meoh_topology) -> openmm.System:
    return tip4p_meoh_de_force_field.create_interchange(aq_meoh_topology).to_openmm(
        combine_nonbonded_forces=False
    )


@pytest.fixture()
def aq_nacl_lj_nonbonded(aq_nacl_lj_system) -> openmm.NonbondedForce:
    nonbonded_forces = [
        force
        for force in aq_nacl_lj_system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ]

    assert len(nonbonded_forces) == 1
    assert (
        len(
            [
                force
                for force in aq_nacl_lj_system.getForces()
                if isinstance(force, openmm.CustomNonbondedForce)
            ]
        )
        == 0
    )

    return nonbonded_forces[0]


@pytest.fixture()
def aq_meoh_de_nonbonded(aq_meoh_de_system) -> openmm.CustomNonbondedForce:
    nonbonded_forces = [
        force
        for force in aq_meoh_de_system.getForces()
        if isinstance(force, openmm.CustomNonbondedForce)
    ]
    assert len(nonbonded_forces) == 1

    return nonbonded_forces[0]


@pytest.fixture()
def argon_force_field() -> openff.toolkit.ForceField:
    force_field = openff.toolkit.ForceField()

    force_field.get_parameter_handler("Electrostatics")
    force_field.get_parameter_handler(
        "ChargeIncrementModel",
        {"version": "0.3", "partial_charge_method": "formal_charge"},
    )

    vdw_handler = force_field.get_parameter_handler("vdW")
    vdw_handler.add_parameter(
        {
            "smirks": "[#18:1]",
            "epsilon": 125.7 * openff.units.unit.kelvin * MOLAR_GAS_CONSTANT_R,
            "sigma": 0.3345 * openff.units.unit.nanometers,
        }
    )

    return force_field


@pytest.fixture()
def argon_eq_schema():
    return Config(
        system=System(solutes={"[#18]": 1}, solvent_a={"[#18]": 255}, solvent_b=None),
        temperature=85.5 * openmm.unit.kelvin,
        pressure=1.0 * openmm.unit.atmosphere,
        alchemical_protocol_a=EquilibriumProtocol(
            minimization_protocol=None,
            equilibration_protocol=None,
            production_protocol=SimulationProtocol(
                n_iterations=1, n_steps_per_iteration=1
            ),
            lambda_sterics=[1.0, 0.5, 0.0],
            lambda_electrostatics=[0.0, 0.0, 0.0],
            sampler="hremd",
        ),
        alchemical_protocol_b=EquilibriumProtocol(
            minimization_protocol=None,
            equilibration_protocol=None,
            production_protocol=SimulationProtocol(
                n_iterations=1, n_steps_per_iteration=1
            ),
            lambda_sterics=[1.0, 0.0],
            lambda_electrostatics=[1.0, 1.0],
            sampler="hremd",
        ),
    )


@pytest.fixture()
def argon_neq_schema():
    return Config(
        system=System(solutes={"[#18]": 1}, solvent_a={"[#18]": 255}, solvent_b=None),
        temperature=85.5 * openmm.unit.kelvin,
        pressure=1.0 * openmm.unit.atmosphere,
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
