import os
import time

import click
from openff.utilities import temporary_cd
from yank.analyze import ExperimentAnalyzer
from yank.experiment import ExperimentBuilder

from absolv.models import DeltaG, TransferFreeEnergyResult, TransferFreeEnergySchema
from absolv.runners.equilibrium import EquilibriumRunner
from absolv.runners.nonequilibrium import NonEquilibriumRunner


def run_absolv(method: str) -> TransferFreeEnergyResult:

    runner_cls = NonEquilibriumRunner if method == "neq" else EquilibriumRunner
    runner_cls.run("", platform="CUDA")

    return runner_cls.analyze("")


def run_yank() -> TransferFreeEnergyResult:

    exp_builder = ExperimentBuilder("yank.yaml")
    exp_builder.run_experiments()

    analyzer = ExperimentAnalyzer("experiments")
    analysed_output = analyzer.auto_analyze()

    result = TransferFreeEnergyResult(
        input_schema=TransferFreeEnergySchema.parse_file("schema.json"),
        delta_g_solvent_a=DeltaG(
            value=analysed_output["free_energy"]["solvent1"]["free_energy_diff"],
            std_error=analysed_output["free_energy"]["solvent1"][
                "free_energy_diff_error"
            ],
        ),
        delta_g_solvent_b=DeltaG(
            value=analysed_output["free_energy"]["solvent2"]["free_energy_diff"],
            std_error=analysed_output["free_energy"]["solvent2"][
                "free_energy_diff_error"
            ],
        ),
    )
    return result


@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.command()
def main(directory):

    *_, method, name = directory.split(os.sep)

    start_time = time.perf_counter()

    with temporary_cd(directory):

        if method != "yank":
            free_energies = run_absolv(method)
        else:
            free_energies = run_yank()

        print(free_energies)

        with open("free-energies.json", "w") as file:
            file.write(free_energies.json(indent=4))

    end_time = time.perf_counter()

    print(f"finished after {(end_time - start_time) / 60.0:.2f}m")


if __name__ == "__main__":
    main()
