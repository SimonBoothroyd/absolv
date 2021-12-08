import os
import time

import click
from openff.utilities import temporary_cd

from absolv.runners.equilibrium import EquilibriumRunner
from absolv.runners.nonequilibrium import NonEquilibriumRunner


@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.command
def main(directory):

    *_, method, name = directory.split(os.sep)

    runner_cls = EquilibriumRunner if method == "eq" else NonEquilibriumRunner

    with temporary_cd(directory):

        start_time = time.perf_counter()
        runner_cls.run("", platform="CUDA")
        end_time = time.perf_counter()

        print(f"finished after {(end_time - start_time) / 60.0:.2f}m")

        free_energies = runner_cls.analyze("")
        print(free_energies)

        with open("free-energies.json", "w") as file:
            file.write(free_energies.json(indent=4))


if __name__ == "__main__":
    main()
