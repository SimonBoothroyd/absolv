import shutil

import numpy
import openmm.unit
import pytest

from absolv.setup import (
    _approximate_box_size_by_density,
    _generate_input_file,
    _molecule_from_smiles,
    setup_system,
)


def test_approximate_box_size_by_density():
    box_size = _approximate_box_size_by_density(
        components=[("O", 128), ("O", 128)],
        target_density=1.0 * openmm.unit.grams / openmm.unit.milliliters,
    )

    assert isinstance(box_size, openmm.unit.Quantity)
    assert box_size.unit.is_compatible(openmm.unit.angstrom)

    box_size = box_size.value_in_unit(openmm.unit.angstrom)

    assert isinstance(box_size, float)

    expected_length = (256.0 * 18.01528 / 6.02214076e23 * 1.0e24) ** (1.0 / 3.0)
    assert numpy.isclose(box_size, expected_length, atol=3)


def test_generate_input_file(monkeypatch):
    monkeypatch.setenv("ABSOLV_PACKMOL_SEED", "1234")

    actual_input_file = _generate_input_file(
        [("CO.xyz", 1)], 1.0 * openmm.unit.angstrom, 0.1 * openmm.unit.nanometers
    )

    expected_input_file = "\n".join(
        [
            "tolerance 1.000000",
            "filetype xyz",
            "output output.xyz",
            "seed 1234",
            "",
            "structure CO.xyz",
            "  number 1",
            "  inside box 0. 0. 0. 1.0 1.0 1.0",
            "end structure",
            "",
        ]
    )

    assert actual_input_file == expected_input_file


@pytest.mark.parametrize(
    "smiles, expected_order",
    [
        ("O", [8, 1, 1]),
        ("[O:1]([H:2])[H:3]", [8, 1, 1]),
        ("[H:2][O:1][H:3]", [8, 1, 1]),
        ("[O:2]([H:1])[H:3]", [1, 8, 1]),
        ("[O:2]([H])[H:3]", [8, 1, 1]),
    ]
)
def test_molecule_from_smiles(smiles, expected_order):
    molecule = _molecule_from_smiles(smiles)

    assert [atom.atomic_number for atom in molecule.atoms] == expected_order



def test_setup_system_packmol_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda *_: None)

    with pytest.raises(FileNotFoundError):
        setup_system([])


def test_setup_system(tmp_cwd):
    assert len(list(tmp_cwd.glob("*"))) == 0

    topology, coordinates = setup_system(
        [("O", 1), ("CO", 2), ("[H]O[H]", 3)],
    )

    assert len(list(tmp_cwd.glob("*"))) == 0

    assert topology.n_molecules == 6
    assert topology.n_unique_molecules == 2

    assert [
        molecule.to_smiles(explicit_hydrogens=False) for molecule in topology.molecules
    ] == ["O", "CO", "CO", "O", "O", "O"]

    assert topology.box_vectors is not None
    assert topology.box_vectors.shape == (3, 3)

    assert isinstance(coordinates, openmm.unit.Quantity)
    assert coordinates.unit.is_compatible(openmm.unit.angstrom)

    coordinates = coordinates.value_in_unit(openmm.unit.angstrom)
    assert isinstance(coordinates, numpy.ndarray)

    assert coordinates.shape == (3 + 6 * 2 + 3 * 3, 3)
    assert not numpy.allclose(coordinates, 0.0)
