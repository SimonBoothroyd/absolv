import shutil
from glob import glob

import numpy
import pytest
from openff.utilities import temporary_cd
from openmm import unit

from absolv.factories.coordinate import PACKMOLCoordinateFactory


class TestPACKMOLCoordinateFactory:
    def test_approximate_box_size_by_density(self):

        box_size = PACKMOLCoordinateFactory._approximate_box_size_by_density(
            components=[("O", 128), ("O", 128)],
            target_density=1.0 * unit.grams / unit.milliliters,
            scale_factor=2.0,
        )

        assert isinstance(box_size, unit.Quantity)
        assert box_size.unit.is_compatible(unit.angstrom)

        box_size = box_size.value_in_unit(unit.angstrom)

        assert isinstance(box_size, float)

        expected_length = (256.0 * 18.01528 / 6.02214076e23 * 1.0e24) ** (
            1.0 / 3.0
        ) * 2.0
        assert numpy.isclose(box_size, expected_length, atol=3)

    def test_build_input_file(self):

        actual_input_file = PACKMOLCoordinateFactory._build_input_file(
            [("CO.xyz", 1), ("O.xyz", 2), ("CCO.xyz", 3)],
            box_size=1.0 * unit.angstrom,
            tolerance=0.1 * unit.nanometers,
        )

        expected_input_file = "\n".join(
            [
                "tolerance 1.000000",
                "filetype xyz",
                "output output.xyz",
                "",
                "structure CO.xyz",
                "  number 1",
                "  inside box 0. 0. 0. 1.0 1.0 1.0",
                "end structure",
                "",
                "structure O.xyz",
                "  number 2",
                "  inside box 0. 0. 0. 1.0 1.0 1.0",
                "end structure",
                "",
                "structure CCO.xyz",
                "  number 3",
                "  inside box 0. 0. 0. 1.0 1.0 1.0",
                "end structure",
                "",
            ]
        )

        assert actual_input_file == expected_input_file

    def test_generate_packmol_missing(self, monkeypatch):

        monkeypatch.setattr(shutil, "which", lambda *_: None)

        with pytest.raises(FileNotFoundError):
            PACKMOLCoordinateFactory.generate([])

    def test_generate(self, tmpdir):

        with temporary_cd(str(tmpdir)):

            assert len(glob("*")) == 0

            topology, coordinates = PACKMOLCoordinateFactory.generate(
                [("O", 1), ("CO", 2), ("[H]O[H]", 3)],
            )

            assert len(glob("*")) == 0

        assert topology.n_topology_molecules == 6
        assert topology.n_reference_molecules == 2

        assert [
            topology_molecule.reference_molecule.to_smiles(explicit_hydrogens=False)
            for topology_molecule in topology.topology_molecules
        ] == ["O", "CO", "CO", "O", "O", "O"]

        assert topology.box_vectors is not None
        assert topology.box_vectors.shape == (3, 3)

        assert isinstance(coordinates, unit.Quantity)
        assert coordinates.unit.is_compatible(unit.angstrom)

        coordinates = coordinates.value_in_unit(unit.angstrom)
        assert isinstance(coordinates, numpy.ndarray)

        assert coordinates.shape == (3 + 6 * 2 + 3 * 3, 3)
        assert not numpy.allclose(coordinates, 0.0)
