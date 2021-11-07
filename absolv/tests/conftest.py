import numpy
import openmm
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import unit


@pytest.fixture(scope="module")
def tip4p_nacl_lj_force_field() -> ForceField:

    force_field = ForceField()

    constraint_handler = force_field.get_parameter_handler("Constraints")
    constraint_handler.add_parameter(
        {"smirks": "[#1:1]-[#8X2H2+0:2]-[#1]", "distance": 0.9572 * unit.angstrom}
    )
    constraint_handler.add_parameter(
        {"smirks": "[#1:1]-[#8X2H2+0]-[#1:2]", "distance": 1.5139 * unit.angstrom}
    )

    vdw_handler = force_field.get_parameter_handler("vdW")
    vdw_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
            "epsilon": 0.0 * unit.kilojoule_per_mole,
            "sigma": 1.0 * unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
            "epsilon": 78.0 * unit.kelvin * unit.MOLAR_GAS_CONSTANT_R,
            "sigma": 3.154 * unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#11+1:1]",
            "epsilon": 0.0874393 * unit.kilocalories_per_mole,
            "rmin_half": 1.369 * unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#17X0-1:1]",
            "epsilon": 0.035591 * unit.kilocalories_per_mole,
            "rmin_half": 2.513 * unit.angstrom,
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
            "smirks": "[#1:1]-[#8X2H2+0:2]-[#1:3]",
            "type": "DivalentLonePair",
            "distance": -0.106 * unit.angstrom,
            "outOfPlaneAngle": 0.0 * unit.degrees,
            "match": "once",
            "charge_increment1": 1.0552 * 0.5 * unit.elementary_charge,
            "charge_increment2": 0.0 * unit.elementary_charge,
            "charge_increment3": 1.0552 * 0.5 * unit.elementary_charge,
        }
    )

    return force_field


@pytest.fixture(scope="module")
def tip4p_meoh_de_force_field() -> ForceField:

    force_field = ForceField(load_plugins=True)

    constraint_handler = force_field.get_parameter_handler("Constraints")
    constraint_handler.add_parameter(
        {"smirks": "[#1:1]-[#8X2H2+0:2]-[#1]", "distance": 0.9572 * unit.angstrom}
    )
    constraint_handler.add_parameter(
        {"smirks": "[#1:1]-[#8X2H2+0]-[#1:2]", "distance": 1.5139 * unit.angstrom}
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
            "epsilon": 0.109 * unit.kilocalories_per_mole,
            "r_min": 3.793 * unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#1:1]",
            "epsilon": 0.0158 * unit.kilocalories_per_mole,
            "r_min": 2.968 * unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#8:1]",
            "epsilon": 0.209 * unit.kilocalories_per_mole,
            "r_min": 3.364 * unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
            "epsilon": 0.0 * unit.kilojoule_per_mole,
            "r_min": 1.0 * unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
            "epsilon": 78.0 * unit.kelvin * unit.MOLAR_GAS_CONSTANT_R,
            "r_min": 3.154 * unit.angstrom,
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
            "charge": [0.0, 0.0, 0.0] * unit.elementary_charge,
        }
    )
    charge_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#6X4:2]-[#8:3]-[#1:4]",
            "charge": [0.0287, 0.1167, -0.5988, 0.396] * unit.elementary_charge,
        }
    )

    virtual_site_handler = force_field.get_parameter_handler("VirtualSites")
    virtual_site_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0:2]-[#1:3]",
            "type": "DivalentLonePair",
            "distance": -0.106 * unit.angstrom,
            "outOfPlaneAngle": 0.0 * unit.degrees,
            "match": "once",
            "charge_increment1": 1.0552 * 0.5 * unit.elementary_charge,
            "charge_increment2": 0.0 * unit.elementary_charge,
            "charge_increment3": 1.0552 * 0.5 * unit.elementary_charge,
        }
    )

    return force_field


@pytest.fixture(scope="module")
def aq_nacl_topology() -> Topology:

    topology = Topology.from_molecules(
        [Molecule.from_smiles("[Na+]"), Molecule.from_smiles("[Cl-]")]
        + [Molecule.from_smiles("O")] * 2
    )
    topology.box_vectors = (numpy.eye(3) * 5.0) * unit.nanometers

    return topology


@pytest.fixture(scope="module")
def aq_meoh_topology() -> Topology:

    return Topology.from_molecules(
        [Molecule.from_smiles("CO")] + [Molecule.from_smiles("O")] * 2
    )


@pytest.fixture(scope="module")
def aq_nacl_lj_system(tip4p_nacl_lj_force_field, aq_nacl_topology) -> openmm.System:
    return tip4p_nacl_lj_force_field.create_openmm_system(aq_nacl_topology)


@pytest.fixture(scope="module")
def aq_meoh_de_system(tip4p_meoh_de_force_field, aq_meoh_topology) -> openmm.System:
    return tip4p_meoh_de_force_field.create_openmm_system(aq_meoh_topology)


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
def argon_force_field() -> ForceField:

    force_field = ForceField()

    force_field.get_parameter_handler("Electrostatics")
    force_field.get_parameter_handler(
        "ChargeIncrementModel",
        {"version": "0.3", "partial_charge_method": "formal_charge"},
    )

    vdw_handler = force_field.get_parameter_handler("vdW")
    vdw_handler.add_parameter(
        {
            "smirks": "[#18:1]",
            "epsilon": 125.7 * unit.kelvin * unit.MOLAR_GAS_CONSTANT_R,
            "sigma": 0.3345 * unit.nanometers,
        }
    )

    return force_field
