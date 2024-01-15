"""Setup systems ready for calculations."""
import errno
import functools
import logging
import os
import random
import shutil
import subprocess

import numpy
import openff.toolkit
import openff.utilities
import openmm

_LOGGER = logging.getLogger(__name__)

_G_PER_ML = openmm.unit.grams / openmm.unit.milliliters


def _approximate_box_size_by_density(
    components: list[tuple[str, int]],
    target_density: openmm.unit.Quantity,
) -> openmm.unit.Quantity:
    """Generate an approximate box size based on the number and molecular weight of
    the molecules present, and a target density for the final system.

    Args:
        components: The list of components.
        target_density: Target mass density for final system with units compatible
            with g / mL.

    Returns:
        The box size.
    """

    molecules = {
        smiles: openff.toolkit.Molecule.from_smiles(smiles)
        for smiles in {smiles for smiles, _ in components}
    }

    volume = 0.0 * openmm.unit.angstrom**3

    for smiles, count in components:
        molecule_mass = (
            functools.reduce(
                (lambda x, y: x + y),
                [atom.mass.to_openmm() for atom in molecules[smiles].atoms],
            )
            / openmm.unit.AVOGADRO_CONSTANT_NA
        )
        molecule_volume = molecule_mass / target_density

        volume += molecule_volume * count

    return volume ** (1.0 / 3.0)


def _generate_input_file(
    components: list[tuple[str, int]],
    box_size: openmm.unit.Quantity,
    tolerance: openmm.unit.Quantity,
) -> str:
    """Generate the PACKMOL input file.

    Args:
        components: The list of components.
        box_size: The size of the box to pack the components into.
        tolerance: The PACKMOL convergence tolerance.

    Returns:
        The string contents of the PACKMOL input file.
    """

    box_size = box_size.value_in_unit(openmm.unit.angstrom)
    tolerance = tolerance.value_in_unit(openmm.unit.angstrom)

    seed = os.getenv("ABSOLV_PACKMOL_SEED")
    seed = seed if seed is not None else random.randint(1, 99999)

    return "\n".join(
        [
            f"tolerance {tolerance:f}",
            "filetype xyz",
            "output output.xyz",
            f"seed {seed}",
            "",
            *[
                f"structure {file_name}\n"
                f"  number {count}\n"
                f"  inside box 0. 0. 0. {box_size} {box_size} {box_size}\n"
                "end structure\n"
                ""
                for file_name, count in components
            ],
        ]
    )


def setup_system(
    components: list[tuple[str, int]],
    box_target_density: openmm.unit.Quantity = 0.95 * _G_PER_ML,
    box_scale_factor: float = 1.1,
    box_padding: openmm.unit.Quantity = 2.0 * openmm.unit.angstrom,
    tolerance: openmm.unit.Quantity = 2.0 * openmm.unit.angstrom,
) -> tuple[openff.toolkit.Topology, openmm.unit.Quantity]:
    """Generate a set of molecule coordinate by using the PACKMOL package.

    Args:
        components: A list of the form ``components[i] = (smiles_i, count_i)`` where
            ``smiles_i`` is the SMILES representation of component `i` and
            ``count_i`` is the number of corresponding instances of that component
            to create.
        box_target_density: Target mass density when approximating the box size for the
            final system with units compatible with g / mL.
        box_scale_factor: The amount to scale the approximate box size by.
        box_padding: The amount of extra padding to add to the box size to avoid PBC
            issues in units compatible with angstroms.
        tolerance: The minimum spacing between molecules during packing in units
             compatible with angstroms.

    Raises:
        * PACKMOLRuntimeError

    Returns:
        A topology containing the molecules the coordinates were generated for and
        a unit [A] wrapped numpy array of coordinates with shape=(n_atoms, 3).
    """

    packmol_path = shutil.which("packmol")

    if packmol_path is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "packmol")

    box_size = (
        _approximate_box_size_by_density(components, box_target_density)
        * box_scale_factor
    )

    molecules = {}

    for smiles, _ in components:
        if smiles in molecules:
            continue

        molecule = openff.toolkit.Molecule.from_smiles(smiles)
        molecule.generate_conformers(n_conformers=1)
        molecule.name = f"component-{len(molecules)}.xyz"
        molecules[smiles] = molecule

    with openff.utilities.temporary_cd():
        for molecule in molecules.values():
            molecule.to_file(molecule.name, "xyz")

        input_file_contents = _generate_input_file(
            [(molecules[smiles].name, count) for smiles, count in components],
            box_size,
            tolerance,
        )

        with open("input.txt", "w") as file:
            file.write(input_file_contents)

        with open("input.txt") as file:
            subprocess.run(packmol_path, stdin=file, check=True, capture_output=True)

        with open("output.xyz") as file:
            output_lines = file.read().splitlines(False)

    coordinates = (
        numpy.array(
            [
                [float(coordinate) for coordinate in coordinate_line.split()[1:]]
                for coordinate_line in output_lines[2:]
                if len(coordinate_line) > 0
            ]
        )
        * openmm.unit.angstrom
    )

    topology = openff.toolkit.Topology.from_molecules(
        [molecules[smiles] for smiles, count in components for _ in range(count)]
    )
    topology.box_vectors = numpy.eye(3) * (box_size + box_padding)

    return topology, coordinates
