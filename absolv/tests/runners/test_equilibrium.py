import os.path
import pickle

import numpy
import openmm
import pytest
from openff.toolkit.topology import Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm.app import PDBFile
from simtk import unit

from absolv.runners.equilibrium import EquilibriumRunner
from absolv.tests import BaseTemporaryDirTest, all_close


class TestEquilibriumRunner(BaseTemporaryDirTest):
    @pytest.mark.parametrize(
        "force_field",
        [
            ForceField("openff-2.0.0.offxml"),
            lambda topology, _: ForceField("openff-2.0.0.offxml").create_openmm_system(
                topology
            ),
        ],
    )
    def test_setup_solvent(self, force_field):

        EquilibriumRunner._setup_solvent([("CO", 1), ("O", 10)], force_field, 1, 10)

        for expected_file in [
            "coords-initial.pdb",
            "coords-initial.npy",
            "system-chemical.xml",
            "system-alchemical.xml",
            "topology.pkl",
        ]:

            assert os.path.isfile(expected_file)

        pdb_file = PDBFile("coords-initial.pdb")

        assert all_close(
            numpy.load("coords-initial.npy") * unit.angstrom,
            numpy.array(
                [
                    [value.value_in_unit(unit.nanometers) for value in coordinate]
                    for coordinate in pdb_file.positions
                ]
            )
            * unit.nanometers,
            atol=1.0e-1,
        )

        with open("topology.pkl", "rb") as file:
            topology: Topology = pickle.load(file)

        assert topology.n_topology_molecules == 11

        with open("system-chemical.xml") as file:

            contents = file.read()

            assert "lambda_sterics" not in contents
            openmm.XmlSerializer.deserialize(contents)

        with open("system-alchemical.xml") as file:

            contents = file.read()

            assert "lambda_sterics" in contents
            openmm.XmlSerializer.deserialize(contents)

    def test_setup(self):
        pytest.skip("not implemented")

    def test_run_solvent(self):
        pytest.skip("not implemented")

    def test_run(self):
        pytest.skip("not implemented")

    def test_analyze_solvent(self):
        pytest.skip("not implemented")

    def test_analyze(self):
        pytest.skip("not implemented")
