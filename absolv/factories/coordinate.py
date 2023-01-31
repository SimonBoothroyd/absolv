"""Factories for building system coordintes."""
import errno
import logging
import os
import shutil
import subprocess
from functools import reduce
from typing import List, Tuple

import numpy
from openff.toolkit.topology import Molecule, Topology
from openff.utilities import temporary_cd
from openmm import unit

_logger = logging.getLogger(__name__)


class PACKMOLRuntimeError(RuntimeError):
    """An error raised when PACKMOL fails to execute / converge for some reason."""


class PACKMOLCoordinateFactory:
    """A factory for generating boxes of molecule coordinates using PACKMOL."""

    @classmethod
    def _approximate_box_size_by_density(
        cls,
        components: List[Tuple[str, int]],
        target_density: unit.Quantity,
        scale_factor: float = 1.1,
    ) -> unit.Quantity:
        """Generate an approximate box size based on the number and molecular weight of
        the molecules present, and a target density for the final system.

        Args:
            components: A list of the form ``components[i] = (smiles_i, count_i)`` where
                ``smiles_i`` is the SMILES representation of component `i` and
                ``count_i`` is the number of corresponding instances of that component
                to create.
            target_density: Target mass density for final system with units compatible
                with g / mL.
                If ``None``, ``box_size`` must be provided.
             scale_factor: The amount to scale the box size by.

        Returns:
            The box size.
        """

        molecules = {
            smiles: Molecule.from_smiles(smiles)
            for smiles in {smiles for smiles, _ in components}
        }

        volume = 0.0 * unit.angstrom**3

        for smiles, count in components:

            molecule_mass = (
                reduce(
                    (lambda x, y: x + y),
                    [atom.mass for atom in molecules[smiles].atoms],
                )
                / unit.AVOGADRO_CONSTANT_NA
            )

            molecule_volume = molecule_mass / target_density

            volume += molecule_volume * count

        return volume ** (1.0 / 3.0) * scale_factor

    @classmethod
    def _build_input_file(
        cls,
        components: List[Tuple[str, int]],
        box_size: unit.Quantity,
        tolerance: unit.Quantity,
    ) -> str:
        """Construct the PACKMOL input file.

        Args:
            components: A list of tuples containing the file path to an XYZ file and
                the number of times to include that component in the final system.
            box_size: The size of the box to pack the components into.
            tolerance: The PACKMOL convergence tolerance.

        Returns:
            The string contents of the PACKMOL input file.
        """

        box_size = box_size.value_in_unit(unit.angstrom)
        tolerance = tolerance.value_in_unit(unit.angstrom)

        seed = os.getenv("ABSOLV_PACKMOL_SEED")

        return "\n".join(
            [
                f"tolerance {tolerance:f}",
                "filetype xyz",
                "output output.xyz",
                *([] if seed is None else [f"seed {seed}"]),
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

    @classmethod
    def generate(
        cls,
        components: List[Tuple[str, int]],
        target_density: unit.Quantity = 0.95 * unit.grams / unit.milliliters,
        tolerance: unit.Quantity = 2.0 * unit.angstrom,
    ) -> Tuple[Topology, unit.Quantity]:
        """Generate a set of molecule coordinate by using the PACKMOL package.

        Args:
            components: A list of the form ``components[i] = (smiles_i, count_i)`` where
                ``smiles_i`` is the SMILES representation of component `i` and
                ``count_i`` is the number of corresponding instances of that component
                to create.
            target_density: Target mass density for final system with units compatible
                with g / mL.
                If ``None``, ``box_size`` must be provided.
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

        box_size = cls._approximate_box_size_by_density(components, target_density)

        molecules = {}

        for smiles, _ in components:

            if smiles in molecules:
                continue

            molecule: Molecule = Molecule.from_smiles(smiles)
            molecule.generate_conformers(n_conformers=1)
            molecule.name = f"component-{len(molecules)}.xyz"
            molecules[smiles] = molecule

        with temporary_cd():

            for molecule in molecules.values():
                molecule.to_file(molecule.name, "xyz")

            input_file_contents = cls._build_input_file(
                [(molecules[smiles].name, count) for smiles, count in components],
                box_size,
                tolerance,
            )

            with open("input.txt", "w") as file:
                file.write(input_file_contents)

            with open("input.txt") as file:
                result = subprocess.check_output(
                    packmol_path, stdin=file, stderr=subprocess.STDOUT
                ).decode("utf-8")

            if not result.find("Success!") > 0:
                raise PACKMOLRuntimeError(result)

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
            * unit.angstrom
        )

        # Add a 2 angstrom buffer to help alleviate PBC issues.
        box_vectors = numpy.eye(3) * (box_size.value_in_unit(unit.angstrom) + 2.0)

        topology = Topology.from_molecules(
            [molecules[smiles] for smiles, count in components for _ in range(count)]
        )
        topology.box_vectors = box_vectors * unit.angstrom

        return topology, coordinates
