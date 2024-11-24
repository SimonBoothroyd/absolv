"""Run calculations defined by a config."""

import collections
import functools
import multiprocessing
import pathlib
import tempfile
import typing

import femto.fe.ddg
import femto.md.config
import femto.md.constants
import femto.md.hremd
import femto.md.simulate
import femto.md.utils.openmm
import mdtraj
import numpy
import openff.toolkit
import openff.utilities
import openmm
import openmm.app
import openmm.unit
import pymbar
import tqdm

import absolv.config
import absolv.fep
import absolv.neq
import absolv.setup
import absolv.utils.openmm
import absolv.utils.topology


class PreparedSystem(typing.NamedTuple):
    """A container for the prepared inputs for a particular solvent phase."""

    system: openmm.System
    """The alchemically modified OpenMM system."""

    topology: openmm.app.Topology
    """The OpenMM topology with any box vectors set."""
    coords: openmm.unit.Quantity
    """The coordinates of the system."""


def _rebuild_topology(
    orig_top: openff.toolkit.Topology,
    orig_coords: openmm.unit.Quantity,
    system: openmm.System,
) -> tuple[openmm.app.Topology, openmm.unit.Quantity, list[set[int]]]:
    """Rebuild the topology to also include virtual sites."""
    atom_idx_to_residue_idx = {}
    atom_idx = 0

    for residue_idx, molecule in enumerate(orig_top.molecules):
        for _ in molecule.atoms:
            atom_idx_to_residue_idx[atom_idx] = residue_idx
            atom_idx += 1

    particle_idx_to_atom_idx = {}
    atom_idx = 0

    for particle_idx in range(system.getNumParticles()):
        if system.isVirtualSite(particle_idx):
            continue

        particle_idx_to_atom_idx[particle_idx] = atom_idx
        atom_idx += 1

    atoms_off = [*orig_top.atoms]
    particles = []

    for particle_idx in range(system.getNumParticles()):
        if system.isVirtualSite(particle_idx):
            v_site = system.getVirtualSite(particle_idx)

            parent_idxs = {
                particle_idx_to_atom_idx[v_site.getParticle(i)]
                for i in range(v_site.getNumParticles())
            }
            parent_residue = atom_idx_to_residue_idx[next(iter(parent_idxs))]

            particles.append((-1, parent_residue))
            continue

        atom_idx = particle_idx_to_atom_idx[particle_idx]
        residue_idx = atom_idx_to_residue_idx[atom_idx]

        particles.append((atoms_off[atom_idx].atomic_number, residue_idx))

    topology = openmm.app.Topology()

    if orig_top.box_vectors is not None:
        topology.setPeriodicBoxVectors(orig_top.box_vectors.to_openmm())

    chain = topology.addChain()

    atom_counts_per_residue = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    atoms = []

    last_residue_idx = -1
    residue = None

    residue_to_particle_idx = collections.defaultdict(list)

    for particle_idx, (atomic_num, residue_idx) in enumerate(particles):
        if residue_idx != last_residue_idx:
            last_residue_idx = residue_idx
            residue = topology.addResidue("UNK", chain)

        element = (
            None if atomic_num < 0 else openmm.app.Element.getByAtomicNumber(atomic_num)
        )
        symbol = "X" if element is None else element.symbol

        atom_counts_per_residue[residue_idx][atomic_num] += 1
        atom = topology.addAtom(
            f"{symbol}{atom_counts_per_residue[residue_idx][atomic_num]}".ljust(3, "x"),
            element,
            residue,
        )
        atoms.append(atom)

        residue_to_particle_idx[residue_idx].append(particle_idx)

    _rename_residues(topology)

    atom_idx_to_particle_idx = {j: i for i, j in particle_idx_to_atom_idx.items()}

    for bond in orig_top.bonds:
        if atoms[atom_idx_to_particle_idx[bond.atom1_index]].residue.name == "HOH":
            continue

        topology.addBond(
            atoms[atom_idx_to_particle_idx[bond.atom1_index]],
            atoms[atom_idx_to_particle_idx[bond.atom2_index]],
        )

    coords_full = []

    for particle_idx in range(system.getNumParticles()):
        if particle_idx in particle_idx_to_atom_idx:
            coords_i = orig_coords[particle_idx_to_atom_idx[particle_idx]]
            coords_full.append(coords_i.value_in_unit(openmm.unit.angstrom))
        else:
            coords_full.append(numpy.zeros((1, 3)))

    coords_full = numpy.vstack(coords_full) * openmm.unit.angstrom

    if len(orig_coords) != len(coords_full):
        context = openmm.Context(system, openmm.VerletIntegrator(1.0))
        context.setPositions(coords_full)
        context.computeVirtualSites()

        coords_full = context.getState(getPositions=True).getPositions(asNumpy=True)

    residues = [
        set(residue_to_particle_idx[residue_idx])
        for residue_idx in range(len(residue_to_particle_idx))
    ]

    return topology, coords_full, residues


def _rename_residues(topology: openmm.app.Topology):
    """Attempts to assign standard residue names to known residues"""

    for residue in topology.residues():
        symbols = sorted(
            (
                atom.element.symbol
                for atom in residue.atoms()
                if atom.element is not None
            )
        )

        if symbols == ["H", "H", "O"]:
            residue.name = "HOH"

            for i, atom in enumerate(residue.atoms()):
                atom.name = "OW" if atom.element.symbol == "O" else f"HW{i}"


def _setup_solvent(
    solvent_idx: typing.Literal["solvent-a", "solvent-b"],
    components: list[tuple[str, int]],
    force_field: openff.toolkit.ForceField | absolv.utils.openmm.SystemGenerator,
    n_solute_molecules: int,
    n_solvent_molecules: int,
    custom_alchemical_potential: str | None = None,
) -> PreparedSystem:
    """Creates the input files for a particular solvent phase.

    Args:
        components: The components present in the system.
        force_field: The force field or system generator function.
        n_solute_molecules: The number of solute molecules, assumed to be the first
            entries in ``components``.
        n_solvent_molecules: The number of solvent molecules, assumed to be the last
            entries in ``components``.
        custom_alchemical_potential: The custom chemical-alchemical intermolecular
            interaction expression.
    """

    n_total_molecules = sum(c for _, c in components)
    assert n_total_molecules == n_solute_molecules + n_solvent_molecules

    is_vacuum = n_solvent_molecules == 0

    topology_off, coords = absolv.setup.setup_system(components)
    topology_off.box_vectors = None if is_vacuum else topology_off.box_vectors

    if isinstance(force_field, openff.toolkit.ForceField):
        original_system = force_field.create_openmm_system(topology_off)
    else:
        original_system: openmm.System = force_field(topology_off, coords, solvent_idx)

    topology, coords, atom_indices = _rebuild_topology(
        topology_off, coords, original_system
    )

    alchemical_indices = atom_indices[:n_solute_molecules]
    persistent_indices = atom_indices[n_solute_molecules:]

    alchemical_system = absolv.fep.apply_fep(
        original_system,
        alchemical_indices,
        persistent_indices,
        custom_alchemical_potential,
    )
    return PreparedSystem(alchemical_system, topology, coords)


def setup(
    system: absolv.config.System,
    config: absolv.config.Config,
    force_field: openff.toolkit.ForceField | absolv.utils.openmm.SystemGenerator,
    custom_alchemical_potential: str | None = None,
) -> tuple[PreparedSystem, PreparedSystem]:
    """Prepare each system to be simulated, namely the ligand in each solvent.

    Args:
        system: The system to prepare.
        config: The config defining the calculation to perform.
        force_field: The force field, or a callable that transforms an OpenFF
            topology into an OpenMM system, **without** any alchemical modifications
            to run the calculations using.

            If a callable is specified, it should take arguments of an OpenFF
            topology, a unit wrapped numpy array of atom coords, and a string
            literal with a value of either ``"solvent-a"`` or ``"solvent-b"``.
        custom_alchemical_potential: A custom expression to use for the potential
            energy function that describes the chemical-alchemical intermolecular
            interactions.

            See the ``absolv.fep.apply_fep`` function for more details.

    Returns:
        The two prepared systems, corresponding to solvent-a and solvent-b respectively.
    """

    solvated_a = _setup_solvent(
        "solvent-a",
        system.components_a,
        force_field,
        system.n_solute_molecules,
        system.n_solvent_molecules_a,
        custom_alchemical_potential,
    )
    solvated_b = _setup_solvent(
        "solvent-b",
        system.components_b,
        force_field,
        system.n_solute_molecules,
        system.n_solvent_molecules_b,
        custom_alchemical_potential,
    )

    if system.solvent_a is not None and config.pressure is not None:
        absolv.utils.openmm.add_barostat(
            solvated_a.system, config.temperature, config.pressure
        )
    if system.solvent_b is not None and config.pressure is not None:
        absolv.utils.openmm.add_barostat(
            solvated_b.system, config.temperature, config.pressure
        )

    return solvated_a, solvated_b


def _equilibrate(
    prepared_system: PreparedSystem,
    temperature: openmm.unit.Quantity,
    minimization_protocol: absolv.config.MinimizationProtocol,
    equilibration_protocol: absolv.config.SimulationProtocol,
    platform: femto.md.constants.OpenMMPlatform,
) -> openmm.State:
    integrator = femto.md.utils.openmm.create_integrator(
        equilibration_protocol.integrator, temperature
    )

    simulation = absolv.utils.openmm.create_simulation(
        prepared_system.system,
        prepared_system.topology,
        prepared_system.coords,
        integrator,
        platform,
    )
    absolv.fep.set_fep_lambdas(simulation.context, 1.0, 1.0)

    simulation.minimizeEnergy(
        minimization_protocol.tolerance, minimization_protocol.max_iterations
    )
    simulation.step(equilibration_protocol.n_steps)

    return simulation.context.getState(getPositions=True)


def _run_eq_phase(
    protocol: absolv.config.EquilibriumProtocol,
    prepared_system: PreparedSystem,
    output_dir: pathlib.Path | None,
    temperature: openmm.unit.Quantity,
    platform: femto.md.constants.OpenMMPlatform,
) -> tuple[dict[str, float], dict[str, numpy.ndarray]]:
    """Run HREMD for one of the solvent phases.

    Args:
        protocol: The protocol to run.
        prepared_system: The prepared system to run.
        output_dir: The (optional) directory to save HREMD samples to.
        temperature: The temperature to run at.
        platform: The OpenMM platform to run using.

    Returns:
        The free energy estimates and the overlap matrices. See
        ``femto.fe.ddg.estimate_ddg`` for more details.
    """
    platform = (
        femto.md.constants.OpenMMPlatform.REFERENCE
        if prepared_system.topology.getPeriodicBoxVectors() is None
        else platform
    )

    equilibrated_coords = _equilibrate(
        prepared_system,
        temperature,
        protocol.minimization_protocol,
        protocol.equilibration_protocol,
        platform,
    )

    integrator = femto.md.utils.openmm.create_integrator(
        protocol.production_protocol.integrator, temperature
    )
    simulation = absolv.utils.openmm.create_simulation(
        prepared_system.system,
        prepared_system.topology,
        prepared_system.coords,
        integrator,
        platform,
    )
    simulation.context.setState(equilibrated_coords)

    states = [
        {
            absolv.fep.LAMBDA_ELECTROSTATICS: lambda_electrostatics,
            absolv.fep.LAMBDA_STERICS: lambda_sterics,
        }
        for lambda_electrostatics, lambda_sterics in zip(
            protocol.lambda_electrostatics, protocol.lambda_sterics, strict=True
        )
    ]
    hremd_config = femto.md.config.HREMD(
        temperature=temperature,
        n_warmup_steps=protocol.production_protocol.n_warmup_steps,
        n_steps_per_cycle=protocol.production_protocol.n_steps_per_cycle,
        n_cycles=protocol.production_protocol.n_cycles,
        trajectory_interval=protocol.production_protocol.trajectory_interval,
        trajectory_enforce_pbc=protocol.production_protocol.trajectory_enforce_pbc,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = pathlib.Path(tmp_dir) if output_dir is None else output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        femto.md.hremd.run_hremd(simulation, states, hremd_config, output_dir)
        u_kn, n_k = femto.fe.ddg.load_u_kn(output_dir / "samples.arrow")

    return femto.fe.ddg.estimate_ddg(u_kn, n_k, temperature)


def run_eq(
    config: absolv.config.Config,
    prepared_system_a: PreparedSystem,
    prepared_system_b: PreparedSystem,
    platform: femto.md.constants.OpenMMPlatform = femto.md.constants.OpenMMPlatform.CUDA,
    output_dir: pathlib.Path | None = None,
    parallel: bool = False,
) -> absolv.config.Result:
    """Perform a simulation at each lambda window and for each solvent.

    Args:
        config: The config defining the calculation to perform.
        prepared_system_a: The prepared system a. See ``setup`` for more details.
        prepared_system_b: The prepared system b. See ``setup`` for more details.
        platform: The OpenMM platform to run using.
        output_dir: The (optional) directory to save HREMD samples to.
        parallel: Whether to run calculations for solvent A and solvent B in
            parallel. This is mostly useful when running HFE calculations where
            the vacuum phase will typically run on the CPU while the solvent phase
            will run on the GPU.
    """

    output_dir_a = None if output_dir is None else output_dir / "solvent-a"
    output_dir_b = None if output_dir is None else output_dir / "solvent-b"

    args = [
        (config.alchemical_protocol_a, prepared_system_a, output_dir_a),
        (config.alchemical_protocol_b, prepared_system_b, output_dir_b),
    ]
    run_fn = functools.partial(
        _run_eq_phase, temperature=config.temperature, platform=platform
    )

    if parallel:
        with multiprocessing.Pool(2) as pool:
            results = list(pool.starmap(run_fn, args))
    else:
        results = [run_fn(*args[0]), run_fn(*args[1])]

    results_a, overlap_a = results[0]
    results_b, overlap_b = results[1]

    dg_a, dg_a_std = results_a["ddG_kcal_mol"], results_a["ddG_error_kcal_mol"]
    # overlap_a = overlap_a["overlap_0"]
    dg_b, dg_b_std = results_b["ddG_kcal_mol"], results_b["ddG_error_kcal_mol"]
    # overlap_b = overlap_b["overlap_0"]

    return absolv.config.Result(
        dg_solvent_a=dg_a * openmm.unit.kilocalorie_per_mole,
        dg_std_solvent_a=dg_a_std * openmm.unit.kilocalorie_per_mole,
        dg_solvent_b=dg_b * openmm.unit.kilocalorie_per_mole,
        dg_std_solvent_b=dg_b_std * openmm.unit.kilocalorie_per_mole,
    )


def _run_phase_end_states(
    protocol: absolv.config.NonEquilibriumProtocol,
    temperature: openmm.unit.Quantity,
    prepared_system: PreparedSystem,
    output_dir: pathlib.Path,
    platform: femto.md.constants.OpenMMPlatform,
):
    platform = (
        femto.md.constants.OpenMMPlatform.REFERENCE
        if prepared_system.topology.getPeriodicBoxVectors() is None
        else platform
    )

    equilibrated_coords = _equilibrate(
        prepared_system,
        temperature,
        protocol.minimization_protocol,
        protocol.equilibration_protocol,
        platform,
    )

    integrator = femto.md.utils.openmm.create_integrator(
        protocol.production_protocol.integrator, temperature
    )
    simulation = absolv.utils.openmm.create_simulation(
        prepared_system.system,
        prepared_system.topology,
        prepared_system.coords,
        integrator,
        platform,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (lambda_electrostatics, lambda_sterics) in enumerate(
        [(1.0, 1.0), (0.0, 0.0)]
    ):
        simulation.context.setState(equilibrated_coords)

        reporter = openmm.app.DCDReporter(
            str(output_dir / f"state-{i}.dcd"), protocol.production_report_interval
        )
        simulation.reporters.clear()
        simulation.reporters.append(reporter)

        absolv.fep.set_fep_lambdas(
            simulation.context, lambda_electrostatics, lambda_sterics
        )
        simulation.step(protocol.production_protocol.n_steps)


def _run_switching(
    protocol: absolv.config.NonEquilibriumProtocol,
    temperature: openmm.unit.Quantity,
    prepared_system: PreparedSystem,
    output_dir: pathlib.Path,
    platform: femto.md.constants.OpenMMPlatform,
):
    platform = (
        femto.md.constants.OpenMMPlatform.REFERENCE
        if prepared_system.topology.getPeriodicBoxVectors() is None
        else platform
    )

    mdtraj_topology = mdtraj.Topology.from_openmm(prepared_system.topology)

    trajectory_0 = mdtraj.load_dcd(str(output_dir / "state-0.dcd"), mdtraj_topology)
    trajectory_1 = mdtraj.load_dcd(str(output_dir / "state-1.dcd"), mdtraj_topology)

    assert len(trajectory_0) == len(
        trajectory_1
    ), "trajectories ran in the two end states must have the same length"

    forward_work = numpy.zeros(len(trajectory_0))
    reverse_work = numpy.zeros(len(trajectory_0))

    integrator = femto.md.utils.openmm.create_integrator(
        protocol.production_protocol.integrator, temperature
    )
    simulation = absolv.utils.openmm.create_simulation(
        prepared_system.system,
        prepared_system.topology,
        prepared_system.coords,
        integrator,
        platform,
    )

    for frame_idx in tqdm.tqdm(range(len(trajectory_0)), desc=" NEQ frame"):
        coords_0 = absolv.utils.openmm.extract_frame(trajectory_0, frame_idx)
        coords_1 = absolv.utils.openmm.extract_frame(trajectory_1, frame_idx)

        forward_work[frame_idx], reverse_work[frame_idx] = absolv.neq.run_neq(
            simulation, coords_0, coords_1, protocol.switching_protocol
        )

    bar_result = pymbar.bar(forward_work, reverse_work)
    value, std_error = bar_result["Delta_f"], bar_result["dDelta_f"]

    kt = openmm.unit.MOLAR_GAS_CONSTANT_R * temperature

    return kt * value, kt * std_error


def run_neq(
    config: absolv.config.Config,
    prepared_system_a: PreparedSystem,
    prepared_system_b: PreparedSystem,
    platform: femto.md.constants.OpenMMPlatform = femto.md.constants.OpenMMPlatform.CUDA,
) -> absolv.config.Result:
    """Performs the simulations required to estimate the free energy using a
    non-equilibrium method.

    These include **equilibrium** simulations at the two end states (i.e. fully
    interacting and fully de-coupled solute) for each solvent followed by
    non-equilibrium switching simulations between each end state to compute the
    forward and reverse work values.

    Args:
        config: The config defining the calculation to perform.
        prepared_system_a: The prepared system a. See ``setup`` for more details.
        prepared_system_b: The prepared system b. See ``setup`` for more details.
        platform: The OpenMM platform to run using.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = pathlib.Path(tmp_dir)

        solvent_a_dir = tmp_dir / "solvent-a"
        solvent_b_dir = tmp_dir / "solvent-b"

        _run_phase_end_states(
            config.alchemical_protocol_a,
            config.temperature,
            prepared_system_a,
            solvent_a_dir,
            platform,
        )
        dg_a, dg_std_a = _run_switching(
            config.alchemical_protocol_a,
            config.temperature,
            prepared_system_a,
            solvent_a_dir,
            platform,
        )

        _run_phase_end_states(
            config.alchemical_protocol_b,
            config.temperature,
            prepared_system_b,
            solvent_b_dir,
            platform,
        )
        dg_b, dg_std_b = _run_switching(
            config.alchemical_protocol_b,
            config.temperature,
            prepared_system_b,
            solvent_b_dir,
            platform,
        )

    return absolv.config.Result(
        dg_solvent_a=dg_a.in_units_of(openmm.unit.kilocalories_per_mole),
        dg_std_solvent_a=dg_std_a.in_units_of(openmm.unit.kilocalories_per_mole),
        dg_solvent_b=dg_b.in_units_of(openmm.unit.kilocalories_per_mole),
        dg_std_solvent_b=dg_std_b.in_units_of(openmm.unit.kilocalories_per_mole),
    )
