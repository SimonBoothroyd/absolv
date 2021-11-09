import os.path

import numpy

from absolv.models import TransferFreeEnergyResult
from absolv.runners.equilibrium import EquilibriumRunner
from absolv.tests import BaseTemporaryDirTest


class TestEquilibriumRunner(BaseTemporaryDirTest):
    def test_run(self, argon_eq_schema, argon_force_field):

        EquilibriumRunner.setup(argon_eq_schema, argon_force_field, "test-dir")
        EquilibriumRunner.run("test-dir", platform="Reference")

        assert all(
            os.path.isdir(os.path.join("test-dir", solvent_index, "state-0"))
            for solvent_index in ("solvent-a", "solvent-b")
        )
        assert all(
            os.path.isfile(
                os.path.join(
                    "test-dir", solvent_index, "state-0", "production-trajectory.dcd"
                )
            )
            for solvent_index in ("solvent-a", "solvent-b")
        )

    def test_analyze(self, argon_eq_schema):

        with open("schema.json", "w") as file:
            file.write(argon_eq_schema.json())

        state_indices = {"solvent-a": [0, 1, 2], "solvent-b": [0, 1]}

        for solvent_index in state_indices:

            for i in state_indices[solvent_index]:

                os.makedirs(os.path.join(solvent_index, f"state-{i}"))

                numpy.savetxt(
                    os.path.join(solvent_index, f"state-{i}", "lambda-potentials.csv"),
                    numpy.random.random((1, len(state_indices[solvent_index]))),
                )

        result = EquilibriumRunner.analyze("")

        assert isinstance(result, TransferFreeEnergyResult)
        assert result.input_schema.json() == argon_eq_schema.json()
