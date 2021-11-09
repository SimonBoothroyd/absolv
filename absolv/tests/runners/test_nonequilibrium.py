import copy
import os
import pickle

import mdtraj
import numpy
import pytest
from openff.utilities import temporary_cd

from absolv.models import TransferFreeEnergyResult
from absolv.runners.nonequilibrium import NonEquilibriumRunner
from absolv.simulations import NonEquilibriumOpenMMSimulation
from absolv.tests import BaseTemporaryDirTest


class TestNonEquilibriumRunner(BaseTemporaryDirTest):
    @pytest.fixture(autouse=True)
    def _setup_argon(self, _temporary_cd, argon_force_field):

        for solvent_index, n_particles, cell_lengths, cell_angles in (
            ("solvent-a", 128, numpy.ones((2, 3)) * 10.0, numpy.ones((2, 3)) * 90.0),
            ("solvent-b", 1, None, None),
        ):

            os.makedirs(solvent_index)

            with temporary_cd(solvent_index):

                NonEquilibriumRunner._setup_solvent(
                    solvent_index,
                    [("[Ar]", n_particles)],
                    argon_force_field,
                    1,
                    n_particles - 1,
                )

                with open("topology.pkl", "rb") as file:
                    topology = pickle.load(file).to_openmm()

                trajectory = mdtraj.Trajectory(
                    xyz=numpy.zeros((2, n_particles, 3)),
                    topology=mdtraj.Topology.from_openmm(topology),
                    unitcell_lengths=cell_lengths,
                    unitcell_angles=cell_angles,
                )

                for state_index in (0, 1):

                    os.makedirs(f"state-{state_index}")
                    trajectory.save(
                        os.path.join(
                            f"state-{state_index}", "production-trajectory.dcd"
                        )
                    )

    def test_run_switching_checkpoint(self, argon_neq_schema):

        expected_forward_work = numpy.array([1.0, 2.0, 3.0])
        expected_reverse_work = numpy.array([3.0, 2.0, 1.0])

        with temporary_cd("solvent-a"):

            numpy.savetxt("forward-work.csv", expected_forward_work, delimiter=" ")
            numpy.savetxt("reverse-work.csv", expected_reverse_work, delimiter=" ")

            forward_work, reverse_work = NonEquilibriumRunner._run_switching(
                argon_neq_schema.alchemical_protocol_a,
                argon_neq_schema.state,
                "Reference",
            )

        assert numpy.allclose(forward_work, expected_forward_work)
        assert numpy.allclose(reverse_work, expected_reverse_work)

    def test_run(self, argon_neq_schema, monkeypatch):

        argon_neq_schema = copy.deepcopy(argon_neq_schema)

        monkeypatch.setattr(
            NonEquilibriumRunner, "_run_solvent", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            NonEquilibriumOpenMMSimulation,
            "run",
            lambda *args, **kwargs: (numpy.random.random(), numpy.random.random()),
        )

        with open("schema.json", "w") as file:
            file.write(argon_neq_schema.json())

        NonEquilibriumRunner.run("", "Reference")

        for solvent_index in ("solvent-a", "solvent-b"):

            forward_work = numpy.genfromtxt(
                os.path.join(solvent_index, "forward-work.csv"), delimiter=" "
            )
            reverse_work = numpy.genfromtxt(
                os.path.join(solvent_index, "reverse-work.csv"), delimiter=" "
            )

            assert forward_work.shape == (2,)
            assert not numpy.allclose(forward_work, 0.0)

            assert reverse_work.shape == (2,)
            assert not numpy.allclose(reverse_work, 0.0)

    def test_analyze(self, argon_eq_schema):

        with open("schema.json", "w") as file:
            file.write(argon_eq_schema.json())

        expected_forward_work = numpy.array([1.0, 2.0, 3.0])
        expected_reverse_work = numpy.array([3.0, 2.0, 1.0])

        for solvent_index in ("solvent-a", "solvent-b"):

            numpy.savetxt(
                os.path.join(solvent_index, "forward-work.csv"),
                expected_forward_work,
                delimiter=" ",
            )
            numpy.savetxt(
                os.path.join(solvent_index, "reverse-work.csv"),
                expected_reverse_work,
                delimiter=" ",
            )

        result = NonEquilibriumRunner.analyze("")

        assert isinstance(result, TransferFreeEnergyResult)
        assert result.input_schema.json() == argon_eq_schema.json()
