"""Utilities to manipulate OpenMM objects."""
import typing

import femto.md.constants
import femto.md.utils.openmm
import mdtraj
import openff.toolkit
import openmm
import openmm.app
import openmm.unit

SystemGenerator = typing.Callable[
    [
        openff.toolkit.Topology,
        openmm.unit.Quantity,
        typing.Literal["solvent-a", "solvent-b"],
    ],
    openmm.System,
]


def add_barostat(
    system: openmm.System,
    temperature: openmm.unit.Quantity,
    pressure: openmm.unit.Quantity,
    frequency: int = 25,
):
    """Add a barostat to a system in-place.

    Args:
        system: The system to add the barostat to.
        temperature: The temperature to simulate at.
        pressure: The pressure to simulate at.
        frequency: The frequency at which to apply the barostat.
    """

    barostats = [
        force
        for force in system.getForces()
        if isinstance(force, openmm.MonteCarloBarostat)
    ]
    assert len(barostats) == 0, "the system should not already contain a barostat"

    system.addForce(openmm.MonteCarloBarostat(pressure, temperature, frequency))


def create_simulation(
    system: openmm.System,
    topology: openff.toolkit.Topology,
    coords: openmm.unit.Quantity,
    integrator: openmm.Integrator,
    platform: femto.md.constants.OpenMMPlatform,
) -> openmm.app.Simulation:
    """Creates an OpenMM simulation object.

    Args:
        system: The system to simulate
        topology: The topology being simulated.
        coords: The initial coordinates. Box vectors (if any) will be taken from the
            topology.
        integrator: The integrator to evolve the system with.
        platform: The accelerator to run using.

    Returns:
        The created simulation.
    """
    platform_properties = (
        {"Precision": "mixed"} if platform.upper() in ["CUDA", "OPENCL"] else {}
    )
    platform = openmm.Platform.getPlatformByName(platform)

    if topology.box_vectors is not None:
        system.setDefaultPeriodicBoxVectors(*topology.box_vectors.to_openmm())

    simulation = openmm.app.Simulation(
        topology.to_openmm(), system, integrator, platform, platform_properties
    )

    if topology.box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*topology.box_vectors.to_openmm())

    simulation.context.setPositions(coords)
    simulation.context.setVelocitiesToTemperature(integrator.getTemperature())

    return simulation


def create_system_generator(
    force_field: openmm.app.ForceField,
    solvent_a_nonbonded_method: int,
    solvent_b_nonbonded_method: int,
    nonbonded_cutoff: openmm.unit.Quantity = 1.0 * openmm.unit.nanometer,
    constraints: int | None = None,
    rigid_water: bool | None = None,
    remove_cmm_motion: bool = True,
    hydrogen_mass: openmm.unit.Quantity | None = None,
    switch_distance: openmm.unit.Quantity | None = None,
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
        topology: openff.toolkit.Topology,
        coordinates: openmm.unit.Quantity,
        solvent_idx: typing.Literal["solvent-a", "solvent-b"],
    ) -> openmm.System:
        openmm_topology = topology.to_openmm()

        if topology.box_vectors is not None:
            openmm_topology.setPeriodicBoxVectors(topology.box_vectors.to_openmm())

        # We need to fix the special case of water in order for OMM to correctly apply
        # a constraint between H atoms.
        for chain in openmm_topology.chains():
            for residue in chain.residues():
                if len(residue) != 3:
                    continue

                symbols = sorted(atom.element.symbol for atom in residue.atoms())

                if symbols == ["H", "H", "O"]:
                    residue.name = "HOH"

        from openmm.app import Modeller

        modeller = Modeller(
            openmm_topology,
            [
                openmm.Vec3(coordinate[0], coordinate[1], coordinate[2])
                for coordinate in coordinates.value_in_unit(openmm.unit.nanometers)
            ]
            * openmm.unit.nanometers,
        )
        modeller.addExtraParticles(force_field)

        system = force_field.createSystem(
            modeller.getTopology(),
            nonbondedMethod=(
                solvent_a_nonbonded_method
                if solvent_idx == "solvent-a"
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


def extract_frame(trajectory: mdtraj.Trajectory, idx: int) -> openmm.State:
    """Extracts a frame from a trajectory as an OpenMM state object.

    Args:
        trajectory: The trajectory to extract the frame from.
        idx: The index of the frame to extract.

    Returns:
        The extracted frame.
    """

    system = openmm.System()

    for _ in range(trajectory.n_atoms):
        system.addParticle(1.0 * openmm.unit.dalton)

    context = openmm.Context(system, openmm.VerletIntegrator(0.0001))
    context.setPositions(trajectory.openmm_positions(idx))

    if trajectory.unitcell_vectors is not None:
        context.setPeriodicBoxVectors(*trajectory.openmm_boxes(idx))

    return context.getState(getPositions=True)
