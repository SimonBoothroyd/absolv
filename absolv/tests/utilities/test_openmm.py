import copy

import numpy.random
import openmm
import pytest
from openmm import unit

from absolv.factories.coordinate import PACKMOLCoordinateFactory
from absolv.tests import all_close, is_close
from absolv.utilities.openmm import (
    build_context,
    evaluate_energy,
    extract_positions,
    minimize,
    set_positions,
)
from absolv.utilities.topology import topology_to_components


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

    positions = (
        numpy.random.random((aq_nacl_topology.n_topology_atoms, 3)) * unit.angstrom
    )

    original_system = copy.deepcopy(aq_nacl_lj_system)

    context = build_context(
        aq_nacl_lj_system,
        positions,
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
    state_positions = state.getPositions(asNumpy=True)

    assert all_close(state_positions[: aq_nacl_topology.n_topology_atoms], positions)


def test_get_set_positions(aq_nacl_topology, aq_nacl_lj_system):

    expected_positions = (
        numpy.zeros((aq_nacl_lj_system.getNumParticles(), 3)) * unit.angstrom
    )
    expected_box_vectors = aq_nacl_topology.box_vectors

    context = build_context(
        aq_nacl_lj_system,
        expected_positions,
        expected_box_vectors,
        123.0 * unit.kelvin,
        None,
    )

    current_positions, current_box_vectors = extract_positions(context)

    assert all_close(current_positions, expected_positions)
    assert all_close(current_box_vectors, expected_box_vectors)

    _, expected_positions = PACKMOLCoordinateFactory.generate(
        topology_to_components(aq_nacl_topology),
    )
    assert len(expected_positions) == aq_nacl_topology.n_topology_atoms

    set_positions(context, expected_positions, expected_box_vectors)

    current_positions, current_box_vectors = extract_positions(context)
    assert len(current_positions) == aq_nacl_lj_system.getNumParticles()

    assert all_close(
        current_positions[: aq_nacl_topology.n_topology_atoms], expected_positions
    )
    assert not all_close(
        current_positions[aq_nacl_topology.n_topology_atoms :], 0.0 * unit.angstrom
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

    positions, _ = extract_positions(context)

    delta = positions[0, :] - positions[1, :]
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
