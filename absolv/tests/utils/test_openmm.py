import numpy.random
import openmm
import openmm.app
import openmm.unit
import openff.toolkit

from absolv.tests import is_close
from absolv.utils.openmm import create_system_generator, add_barostat


def test_add_barostat():
    expected_pressure = 2.0 * openmm.unit.bar
    expected_temperature = 200.0 * openmm.unit.kelvin

    system = openmm.System()
    add_barostat(system, expected_temperature, expected_pressure)

    assert system.getNumForces() == 1

    barostat = next(iter(system.getForces()))
    assert isinstance(barostat, openmm.MonteCarloBarostat)

    assert is_close(barostat.getDefaultPressure(), expected_pressure)
    assert is_close(barostat.getDefaultTemperature(), expected_temperature)


def test_create_system_generator(test_data_dir):
    force_field = openmm.app.ForceField(
        str(test_data_dir / "force-fields" / "tip3p.xml"),
        str(test_data_dir / "force-fields" / "methylindole.xml"),
    )

    system_generator = create_system_generator(
        force_field, openmm.app.PME, openmm.app.NoCutoff
    )

    expected_box_length = 30.21 * openmm.unit.angstrom

    topology = openff.toolkit.Topology.from_molecules(
        [
            openff.toolkit.Molecule.from_smiles("CC1=CC2=CC=CC=C2N1"),
            openff.toolkit.Molecule.from_smiles("O"),
        ]
    )
    topology.box_vectors = numpy.eye(3) * expected_box_length

    coords = numpy.zeros((topology.n_atoms, 3)) * openmm.unit.angstrom

    system_a = system_generator(topology, coords, "solvent-a")
    assert isinstance(system_a, openmm.System)
    assert is_close(system_a.getDefaultPeriodicBoxVectors()[0][0], expected_box_length)
    # Make sure the water constraints are properly applied
    assert system_a.getNumConstraints() == 3

    nonbonded_force_a = [
        force
        for force in system_a.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ][0]
    assert nonbonded_force_a.getNonbondedMethod() == openmm.NonbondedForce.PME

    topology.box_vectors = None

    system_b = system_generator(topology, coords, "solvent-b")
    assert isinstance(system_b, openmm.System)
    assert is_close(
        system_b.getDefaultPeriodicBoxVectors()[0][0], 20.0 * openmm.unit.angstrom
    )
    assert system_b.getNumConstraints() == 3

    nonbonded_force_b = [
        force
        for force in system_b.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ][0]
    assert nonbonded_force_b.getNonbondedMethod() == openmm.NonbondedForce.NoCutoff
