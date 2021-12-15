import os.path
import pickle

import numpy
import openmm
import pytest
from openff.toolkit.topology import Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.utilities import temporary_cd
from openmm.app import PDBFile
from simtk import unit

from absolv.models import State
from absolv.runners._runners import BaseRunner
from absolv.simulations import (
    AlchemicalOpenMMSimulation,
    RepexAlchemicalOpenMMSimulation,
)
from absolv.tests import BaseTemporaryDirTest, all_close


class TestBaseRunner(BaseTemporaryDirTest):
    def test_load_solvent_inputs(self, argon_force_field):

        os.makedirs("test-dir")

        with temporary_cd("test-dir"):
            BaseRunner._setup_solvent(
                "solvent-a", [("[Ar]", 10)], argon_force_field, 1, 9
            )

        (
            topology,
            coordinates,
            chemical_system,
            alchemical_system,
        ) = BaseRunner._load_solvent_inputs("test-dir")

        assert isinstance(topology, Topology)

        assert isinstance(coordinates, unit.Quantity)
        assert coordinates.shape == (10, 3)

        assert isinstance(chemical_system, openmm.System)
        assert "lambda" not in openmm.XmlSerializer.serialize(chemical_system)

        assert isinstance(alchemical_system, openmm.System)
        assert "lambda" in openmm.XmlSerializer.serialize(alchemical_system)

    @pytest.mark.parametrize(
        "force_field",
        [
            ForceField("openff-2.0.0.offxml"),
            lambda topology, *_: ForceField("openff-2.0.0.offxml").create_openmm_system(
                topology
            ),
        ],
    )
    def test_setup_solvent(self, force_field):

        BaseRunner._setup_solvent(
            "solvent-a", [("CO", 1), ("O", 10)], force_field, 1, 10
        )

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

    def test_setup(self, argon_eq_schema, argon_force_field):

        BaseRunner.setup(argon_eq_schema, argon_force_field, "argon-dir")

        assert os.path.isdir("argon-dir")

        for solvent_index in ("solvent-a", "solvent-b"):

            assert os.path.isdir(os.path.join("argon-dir", solvent_index))
            assert os.path.isfile(
                os.path.join("argon-dir", solvent_index, "topology.pkl")
            )

        with open(os.path.join("argon-dir", "schema.json")) as file:
            assert argon_eq_schema.json(indent=4) == file.read()

    @pytest.mark.parametrize("sampler", ["independent", "repex"])
    def test_run_solvent(
        self, sampler, argon_eq_schema, argon_force_field, monkeypatch
    ):

        argon_eq_schema.alchemical_protocol_a.sampler = sampler
        argon_eq_schema.alchemical_protocol_b.sampler = sampler

        run_class = None
        run_directories = []

        def mock_run(self, directory):

            nonlocal run_class
            run_class = type(self)

            run_directories.append(directory)

        monkeypatch.setattr(AlchemicalOpenMMSimulation, "run", mock_run)
        monkeypatch.setattr(RepexAlchemicalOpenMMSimulation, "run", mock_run)

        BaseRunner._setup_solvent(
            "solvent-a", [("[Ar]", 128)], argon_force_field, 1, 127
        )
        BaseRunner._run_solvent(
            argon_eq_schema.alchemical_protocol_a,
            State(temperature=88.5, pressure=1.0),
            "Reference",
            states=[0, 2] if sampler != "repex" else None,
        )

        assert run_directories == (
            ["state-0", "state-2"] if sampler != "repex" else [""]
        )

        expected_run_class = (
            AlchemicalOpenMMSimulation
            if sampler != "repex"
            else RepexAlchemicalOpenMMSimulation
        )
        assert expected_run_class == run_class

    def test_run_solvent_state_error(self, argon_eq_schema, argon_force_field):

        argon_eq_schema.alchemical_protocol_a.sampler = "repex"
        argon_eq_schema.alchemical_protocol_b.sampler = "repex"

        BaseRunner._setup_solvent(
            "solvent-a", [("[Ar]", 128)], argon_force_field, 1, 127
        )

        with pytest.raises(NotImplementedError, match="All lambda states must be run"):

            BaseRunner._run_solvent(
                argon_eq_schema.alchemical_protocol_a,
                State(temperature=88.5, pressure=1.0),
                "Reference",
                states=[0, 2],
            )
