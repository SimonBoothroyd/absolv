import femto.md.constants
import mdtraj
import numpy.random
import openff.toolkit
import openmm
import openmm.app
import openmm.unit

from absolv.tests import is_close
from absolv.utils.openmm import (
    add_barostat,
    create_simulation,
    create_system_generator,
    extract_frame,
)


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


def test_create_simulation():
    system = openmm.System()
    system.addParticle(1.0)

    expected_coords = numpy.array([[1.0, 2.0, 3.0]]) * openmm.unit.angstrom
    expected_box = numpy.eye(3) * 12.3 * openmm.unit.angstrom

    topology = openff.toolkit.Molecule.from_smiles("[Ar]").to_topology()
    topology.box_vectors = expected_box

    integrator = openmm.LangevinIntegrator(0.001, 1.0, 0.001)

    simulation = create_simulation(
        system,
        topology.to_openmm(),
        expected_coords,
        integrator,
        femto.md.constants.OpenMMPlatform.REFERENCE,
    )
    assert isinstance(simulation, openmm.app.Simulation)

    found_coords = simulation.context.getState(getPositions=True).getPositions(
        asNumpy=True
    )
    assert numpy.allclose(
        found_coords.value_in_unit(openmm.unit.angstrom),
        expected_coords.value_in_unit(openmm.unit.angstrom),
    )

    found_box = simulation.context.getState().getPeriodicBoxVectors(asNumpy=True)
    assert numpy.allclose(
        found_box.value_in_unit(openmm.unit.angstrom),
        expected_box.value_in_unit(openmm.unit.angstrom),
    )


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


def test_extract_frame():
    expected_coords = numpy.array([[[1.0, 2.0, 3.0]]]) * openmm.unit.angstrom
    expected_lengths = numpy.array([[1.2, 2.3, 3.4]]) * openmm.unit.angstrom

    expected_box = (
        numpy.array([[[1.2, 0.0, 0.0], [0.0, 2.3, 0.0], [0.0, 0.0, 3.4]]])
        * openmm.unit.angstrom
    )

    topology = openff.toolkit.Molecule.from_smiles("[Ar]").to_topology().to_openmm()

    trajectory = mdtraj.Trajectory(
        expected_coords.value_in_unit(openmm.unit.nanometers),
        mdtraj.Topology.from_openmm(topology),
        unitcell_lengths=expected_lengths.value_in_unit(openmm.unit.nanometers),
        unitcell_angles=numpy.array([[90.0, 90.0, 90.0]]),
    )
    print(trajectory)

    state = extract_frame(trajectory, 0)

    found_coords = state.getPositions(asNumpy=True)
    assert numpy.allclose(
        found_coords.value_in_unit(openmm.unit.angstrom),
        expected_coords.value_in_unit(openmm.unit.angstrom)[0],
    )

    found_box = state.getPeriodicBoxVectors(asNumpy=True)
    assert numpy.allclose(
        found_box.value_in_unit(openmm.unit.angstrom),
        expected_box.value_in_unit(openmm.unit.angstrom)[0],
    )
