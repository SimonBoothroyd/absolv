import copy
from typing import Callable, List, Literal, Optional, Tuple, Union, overload

import numpy
import openmm
import openmm.app
from openff.toolkit.topology import Topology
from openmm import unit

SystemGenerator = Callable[
    [Topology, unit.Quantity, Literal["solvent-a", "solvent-b"]], openmm.System
]

OpenMMPlatform = Literal["Reference", "OpenCL", "CUDA", "CPU"]


@overload
def array_to_vectors(box_vectors: numpy.ndarray) -> List[openmm.Vec3]:
    ...


@overload
def array_to_vectors(box_vectors: unit.Quantity) -> unit.Quantity:
    ...


def array_to_vectors(array):
    """A utility to convert numpy arrays into lists of OpenMM vectors.

    Args:
        array: The array to convert with shape=(n_items, 3)

    Returns:
        A list of three ``Vec3`` objects
    """

    original_units = None if not isinstance(array, unit.Quantity) else array.unit

    if original_units is not None:
        array = array.value_in_unit(original_units)

    vectors = [openmm.Vec3(row[0], row[1], row[2]) for row in array]

    return vectors if original_units is None else vectors * original_units


def build_context(
    system: openmm.System,
    coordinates: unit.Quantity,
    box_vectors: Optional[unit.Quantity],
    temperature: unit.Quantity,
    pressure: Optional[unit.Quantity],
    platform: OpenMMPlatform = "Reference",
    timestep: unit.Quantity = 2.0 * unit.femtoseconds,
    thermostat_friction: unit.Quantity = 1.0 / unit.picosecond,
) -> openmm.Context:
    """Builds a new OpenMM context.

    Args:
        system: The system encoding the potential energy function.
        coordinates: The atom coordinates stored in a unit wrapped array with
            shape=(n_atoms, 3).
        box_vectors: The (optional) box vectors stored in a unit wrapped array with
            shape=(3, 3).
        temperature: The temperature to simulate at.
        pressure: The (optional) pressure to simulate at.
        platform: The platform to simulate on.
        timestep: The default timestep to use.
        thermostat_friction: The default thermostat friction to set.
    """

    system = copy.deepcopy(system)

    assert pressure is None or (
        pressure is not None and box_vectors is not None
    ), "box vectors must be provided when the pressure is specified."

    assert (
        len(
            [
                force
                for force in system.getForces()
                if isinstance(force, openmm.MonteCarloBarostat)
            ]
        )
        == 0
    ), "the system should not already contain a barostat"

    if pressure is not None:
        system.addForce(openmm.MonteCarloBarostat(pressure, temperature, 25))

    integrator = openmm.LangevinIntegrator(temperature, thermostat_friction, timestep)

    context = openmm.Context(
        system,
        integrator,
        openmm.Platform.getPlatformByName(platform),
    )

    set_coordinates(context, coordinates, box_vectors)

    context.setVelocitiesToTemperature(temperature * unit.kelvin)

    return context


def set_alchemical_lambdas(
    context: openmm.Context,
    lambda_sterics: Optional[float] = None,
    lambda_electrostatics: Optional[float] = None,
):
    """Set the values of the alchemical lambdas on an OpenMM context.

    Args:
        context: The context to update.
        lambda_sterics: The (optional) value of the steric lambda.
        lambda_electrostatics: The (optional) value of the electrostatics lambda.
    """

    if lambda_sterics is not None:
        assert 0.0 <= lambda_sterics <= 1.0, "`lambda_sterics` must be between 0 and 1"
        context.setParameter("lambda_sterics", lambda_sterics)

    if lambda_electrostatics is not None:

        assert (
            0.0 <= lambda_electrostatics <= 1.0
        ), "`lambda_electrostatics` must be between 0 and 1"

        context.setParameter("lambda_electrostatics", lambda_electrostatics)


def set_coordinates(
    context: openmm.Context,
    coordinates: unit.Quantity,
    box_vectors: Optional[unit.Quantity],
):
    """Updates an OpenMM context with a new set of coordinates and box vectors.

    Args:
        context: The context to update.
        coordinates: The new coordinates stored in a unit wrapped array with
            shape=(n_atoms, 3).
        box_vectors: The (optional) box vectors stored in a unit wrapped array with
            shape=(3, 3) or list of three ``Vec3`` objects.
    """

    if box_vectors is not None:

        if isinstance(box_vectors.value_in_unit(unit.nanometers), numpy.ndarray):
            box_vectors = array_to_vectors(box_vectors)

        context.setPeriodicBoxVectors(*box_vectors)

    system: openmm.System = context.getSystem()

    if len(coordinates) != system.getNumParticles():

        coordinates = coordinates.value_in_unit(unit.nanometers)

        n_v_sites = sum(
            1 for i in range(system.getNumParticles()) if system.isVirtualSite(i)
        )
        n_atoms = system.getNumParticles() - n_v_sites

        assert len(coordinates) == n_atoms, (
            "coordinates must either have shape=(n_atoms, 3) "
            "or (n_atoms + n_v_sites, 3)"
        )

        full_coordinates = numpy.zeros((system.getNumParticles(), 3))
        counter = 0

        for i in range(system.getNumParticles()):

            if system.isVirtualSite(i):
                continue

            full_coordinates[i] = coordinates[counter]
            counter += 1

        coordinates = full_coordinates * unit.nanometers

    context.setPositions(coordinates)
    context.computeVirtualSites()


def extract_coordinates(
    state: Union[openmm.State, openmm.Context]
) -> Tuple[unit.Quantity, unit.Quantity]:
    """Extracts the current coordinates and box vectors from an OpenMM context.

    Args:
        state: The state (or context) to extract from.

    Returns:
        The current coordinates and box vectors stored in unit wrapped arrays with
        shape=(n_atoms, 3) and shape=(3, 3) respectively.
    """

    if isinstance(state, openmm.Context):
        state = state.getState(getPositions=True)

    box_vectors = state.getPeriodicBoxVectors().value_in_unit(unit.nanometers)

    return (
        state.getPositions(asNumpy=True),
        numpy.array(
            [
                [box_vectors[0].x, box_vectors[0].y, box_vectors[0].z],
                [box_vectors[1].x, box_vectors[1].y, box_vectors[1].z],
                [box_vectors[2].x, box_vectors[2].y, box_vectors[2].z],
            ]
        )
        * unit.nanometers,
    )


def minimize(
    context: openmm.Context,
    tolerance: unit.Quantity = 10 * unit.kilojoules_per_mole / unit.nanometer,
    max_iterations: int = 0,
):
    """Energy minimize an OpenMM context.

    Args:
        context: The context to minimize.
        tolerance: How precisely the energy minimum must be located
        max_iterations: The maximum number of iterations to perform. If this is 0,
            minimization is continued until the results converge.
    """
    openmm.LocalEnergyMinimizer.minimize(context, tolerance, max_iterations)


def evaluate_energy(
    system: openmm.System,
    coordinates: unit.Quantity,
    box_vectors: Optional[unit.Quantity] = None,
    lambda_sterics: Optional[float] = None,
    lambda_electrostatics: Optional[float] = None,
    platform: OpenMMPlatform = "Reference",
) -> unit.Quantity:
    """Evaluates the energy of a given system at a particular set of coordinates.

    Args:
        system: The openmm system that should be used to evaluate the energies.
        coordinates: The coordinates that should be used when evaluating the energies.
        box_vectors: The (optional) periodic box vectors that should be used when
            evaluating the energies.
        lambda_sterics: The value of `lambda_sterics` to evaluate the energies at.
        lambda_electrostatics: The value of `lambda_electrostatics` to evaluate the
            energies at.
        platform: The OpenMM platform to simulate using.

    Returns:
        The energy.
    """

    context = build_context(
        system, coordinates, box_vectors, 1.0 * unit.kelvin, None, platform
    )
    set_alchemical_lambdas(context, lambda_sterics, lambda_electrostatics)

    return context.getState(getEnergy=True).getPotentialEnergy()


def create_system_generator(
    force_field: openmm.app.ForceField,
    solvent_a_nonbonded_method: int,
    solvent_b_nonbonded_method: int,
    nonbonded_cutoff: unit.Quantity = 1.0 * unit.nanometer,
    constraints: Optional[int] = None,
    rigid_water: Optional[bool] = None,
    remove_cmm_motion: bool = True,
    hydrogen_mass: Optional[unit.Quantity] = None,
    switch_distance: Optional[unit.Quantity] = None,
) -> SystemGenerator:
    """Creates a 'system generator' that can be used when setting up an alchemical
    free energy calculation from an OpenMM force field.

    Args:
        force_field: The OpenMM force field to parameterize the topology using.
        solvent_a_nonbonded_method: The non-bonded method to use in solvent a.
        solvent_b_nonbonded_method: The non-bonded method to use in solvent b.
        nonbonded_cutoff: The non-bonded cutoff to use.
        constraints: The type of constraints to apply to the system.
        rigid_water: Whether to force rigid water.
        remove_cmm_motion: Whether to remove any CMM motion.
        hydrogen_mass: The mass to use for hydrogens.
        switch_distance: The switch distance to use.

    Returns:
        A callable that will create an OpenMM system from an OpenFF topology and the
        name of the solvent (i.e. ``"solvent-a"`` or ``"solvent-b"``) the system will
        be used for.
    """

    def system_generator(
        topology: Topology,
        coordinates: unit.Quantity,
        solvent_index: Literal["solvent-a", "solvent-b"],
    ) -> openmm.System:

        openmm_topology = topology.to_openmm()

        if topology.box_vectors is not None:
            openmm_topology.setPeriodicBoxVectors(topology.box_vectors)

        # We need to fix the special case of water in order for OMM to correctly apply
        # a constraint between H atoms.
        for chain in openmm_topology.chains():

            for residue in chain.residues():

                if len(residue) != 3:
                    continue

                symbols = sorted(atom.element.symbol for atom in residue.atoms())

                if symbols != ["H", "H", "O"]:
                    continue

                residue.name = "HOH"

        from openmm.app import Modeller

        modeller = Modeller(
            openmm_topology,
            [
                openmm.Vec3(coordinate[0], coordinate[1], coordinate[2])
                for coordinate in coordinates.value_in_unit(unit.nanometers)
            ]
            * unit.nanometers,
        )
        modeller.addExtraParticles(force_field)

        system = force_field.createSystem(
            modeller.getTopology(),
            nonbondedMethod=(
                solvent_a_nonbonded_method
                if solvent_index == "solvent-a"
                else solvent_b_nonbonded_method
            ),
            nonbondedCutoff=nonbonded_cutoff,
            constraints=constraints,
            rigidWater=rigid_water,
            removeCMMotion=remove_cmm_motion,
            hydrogenMass=hydrogen_mass,
            switchDistance=switch_distance,
        )

        return system

    return system_generator
