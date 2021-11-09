import copy
import os

import numpy.random
import openmm
import openmm.app
import pytest
from openff.toolkit.topology import Molecule, Topology
from openmm import unit
from pkg_resources import resource_filename

from absolv.factories.coordinate import PACKMOLCoordinateFactory
from absolv.tests import all_close, is_close
from absolv.utilities.openmm import (
    array_to_vectors,
    build_context,
    create_system_generator,
    evaluate_energy,
    extract_coordinates,
    minimize,
    set_coordinates,
)
from absolv.utilities.topology import topology_to_components


@pytest.mark.parametrize(
    "input_array",
    [numpy.arange(12).reshape((4, 3)), numpy.arange(12).reshape((4, 3)) * unit.kelvin],
)
def test_array_to_vectors(input_array):

    vectors = array_to_vectors(input_array)

    assert len(vectors) == 4

    for row, vector in zip(input_array, vectors):

        assert isinstance(
            vector,
            (openmm.Vec3 if isinstance(input_array, numpy.ndarray) else unit.Quantity),
        )
        assert all(is_close(vector[i], row[i]) for i in range(3))


@pytest.mark.parametrize(
    "box_vectors, pressure",
    [
        (None, None),
        (numpy.eye(3) * (3.0 + numpy.random.random()) * unit.nanometers, None),
        (
            numpy.eye(3) * (3.0 + numpy.random.random()) * unit.nanometers,
            1.0 * unit.atmosphere,
        ),
    ],
)
def test_build_context(box_vectors, pressure, aq_nacl_topology, aq_nacl_lj_system):

    coordinates = (
        numpy.random.random((aq_nacl_topology.n_topology_atoms, 3)) * unit.angstrom
    )

    original_system = copy.deepcopy(aq_nacl_lj_system)

    context = build_context(
        aq_nacl_lj_system,
        coordinates,
        box_vectors,
        123.0 * unit.kelvin,
        pressure,
    )

    # Make sure the initial system wasn't mutated.
    assert openmm.XmlSerializer.serialize(
        original_system
    ) == openmm.XmlSerializer.serialize(aq_nacl_lj_system)

    barostats = [
        force
        for force in context.getSystem().getForces()
        if isinstance(force, openmm.MonteCarloBarostat)
    ]
    assert len(barostats) == (1 if pressure is not None else 0)

    if pressure is not None:
        assert is_close(barostats[0].getDefaultPressure(), pressure)

    assert is_close(context.getIntegrator().getTemperature(), 123.0 * unit.kelvin)

    state: openmm.State = context.getState(getPositions=True)
    state_coordinates = state.getPositions(asNumpy=True)

    assert all_close(
        state_coordinates[: aq_nacl_topology.n_topology_atoms], coordinates
    )


def test_get_set_coordinates(aq_nacl_topology, aq_nacl_lj_system):

    expected_coordinates = (
        numpy.zeros((aq_nacl_lj_system.getNumParticles(), 3)) * unit.angstrom
    )
    expected_box_vectors = aq_nacl_topology.box_vectors

    context = build_context(
        aq_nacl_lj_system,
        expected_coordinates,
        expected_box_vectors,
        123.0 * unit.kelvin,
        None,
    )

    current_coordinates, current_box_vectors = extract_coordinates(context)

    assert all_close(current_coordinates, expected_coordinates)
    assert all_close(current_box_vectors, expected_box_vectors)

    _, expected_coordinates = PACKMOLCoordinateFactory.generate(
        topology_to_components(aq_nacl_topology),
    )
    assert len(expected_coordinates) == aq_nacl_topology.n_topology_atoms

    set_coordinates(context, expected_coordinates, expected_box_vectors)

    current_coordinates, current_box_vectors = extract_coordinates(context)
    assert len(current_coordinates) == aq_nacl_lj_system.getNumParticles()

    assert all_close(
        current_coordinates[: aq_nacl_topology.n_topology_atoms], expected_coordinates
    )
    assert not all_close(
        current_coordinates[aq_nacl_topology.n_topology_atoms :], 0.0 * unit.angstrom
    )

    assert all_close(current_box_vectors, expected_box_vectors)


def test_minimize():

    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)

    force = openmm.HarmonicBondForce()
    force.addBond(
        0, 1, 1.0 * unit.angstrom, 4.0 * unit.kilojoule_per_mole / unit.angstrom ** 2
    )
    system.addForce(force)

    context = build_context(
        system,
        numpy.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * unit.angstrom,
        None,
        1.0 * unit.kelvin,
        None,
    )
    minimize(context, tolerance=0.1 * unit.kilojoules_per_mole / unit.nanometer)

    coordinates, _ = extract_coordinates(context)

    delta = coordinates[0, :] - coordinates[1, :]
    distance = (delta * delta).sum().sqrt()

    assert is_close(distance, 1.0 * unit.angstrom)


def test_evaluate_energy():

    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)

    force = openmm.HarmonicBondForce()
    force.addBond(
        0, 1, 1.0 * unit.angstrom, 4.0 * unit.kilojoule_per_mole / unit.angstrom ** 2
    )
    system.addForce(force)

    energy = evaluate_energy(
        system,
        numpy.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * unit.angstrom,
        None,
        None,
        None,
    )

    assert is_close(energy, 4.0 / 2.0 * (2.0 - 1.0) ** 2 * unit.kilojoule_per_mole)


def test_create_system():

    force_field = openmm.app.ForceField(
        *(
            resource_filename(
                "absolv", os.path.join("tests", "data", "force-fields", file_name)
            )
            for file_name in ("tip3p.xml", "methylindole.xml")
        )
    )

    system_generator = create_system_generator(
        force_field,
        openmm.app.PME,
        openmm.app.NoCutoff,
    )

    topology = Topology.from_molecules(
        [Molecule.from_smiles("CC1=CC2=CC=CC=C2N1"), Molecule.from_smiles("O")]
    )
    topology.box_vectors = numpy.eye(3) * 30.21 * unit.angstrom

    coordinates = numpy.zeros((topology.n_topology_atoms, 3)) * unit.angstrom

    system_a = system_generator(topology, coordinates, "solvent-a")
    assert isinstance(system_a, openmm.System)
    assert is_close(
        system_a.getDefaultPeriodicBoxVectors()[0][0], 30.21 * unit.angstrom
    )
    # Make sure the water constraints are properly applied
    assert system_a.getNumConstraints() == 3

    nonbonded_force_a = [
        force
        for force in system_a.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ][0]
    assert nonbonded_force_a.getNonbondedMethod() == openmm.NonbondedForce.PME

    topology.box_vectors = None

    system_b = system_generator(topology, coordinates, "solvent-b")
    assert isinstance(system_b, openmm.System)
    assert is_close(system_b.getDefaultPeriodicBoxVectors()[0][0], 20.0 * unit.angstrom)
    assert system_b.getNumConstraints() == 3

    nonbonded_force_b = [
        force
        for force in system_b.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ][0]
    assert nonbonded_force_b.getNonbondedMethod() == openmm.NonbondedForce.NoCutoff
