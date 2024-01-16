import collections
import json
import logging
import pathlib

import click
import numpy
import openmm.unit
from matplotlib import pyplot
from run import DEFAULT_METHODS, DEFAULT_SOLUTES

import absolv.config

KCAL = openmm.unit.kilocalorie_per_mole


def _compute_dg(result: absolv.config.Result) -> float:
    # for the regression test we set up solvent A as vacuum and solvent B as
    # explicit solvent.
    # in both 'solvents' the alchemical path goes from molecule switched on to
    # molecule switched off.
    return (result.dg_solvent_a + -result.dg_solvent_b).value_in_unit(KCAL)


def _parse_results(
    result_dirs: list[pathlib.Path],
) -> dict[str, dict[str, tuple[float, float]]]:
    results = collections.defaultdict(dict)

    for method in DEFAULT_METHODS:
        for solute in DEFAULT_SOLUTES:
            replica_paths = [
                result_path
                for result_dir in result_dirs
                for result_path in result_dir.glob(f"{method}-{solute}-*/result.json")
            ]

            if len(replica_paths) == 0:
                continue

            replicas = [
                absolv.config.Result.model_validate_json(result_path.read_text())
                for result_path in replica_paths
            ]

            dg = numpy.array([_compute_dg(replica) for replica in replicas])
            results[method][solute] = numpy.mean(dg), numpy.std(dg)

    return {**results}


@click.command()
@click.option(
    "--result-dir",
    "result_dirs",
    multiple=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(
        exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
    required=True,
)
def main(result_dirs: list[pathlib.Path], output_path: pathlib.Path):
    logging.basicConfig(level=logging.INFO)

    reference = json.loads(pathlib.Path("results/reference.json").read_text())
    results = _parse_results(result_dirs)

    figure, axis = pyplot.subplots(1, 1, figsize=(4, 4))

    for method in results:
        solutes = sorted(results[method])

        x = [reference[solute]["value"] for solute in solutes]
        x_std = [reference[solute]["std_error"] for solute in solutes]

        y = [results[method][solute][0] for solute in solutes]
        y_std = [results[method][solute][1] for solute in solutes]

        axis.errorbar(
            x, y, xerr=x_std, yerr=y_std, label=method, marker="x", linestyle="none"
        )

    lims = [
        numpy.min([axis.get_xlim(), axis.get_ylim()]),
        numpy.max([axis.get_xlim(), axis.get_ylim()]),
    ]

    # now plot both limits against eachother
    axis.plot(lims, lims, "k-", alpha=0.25, zorder=0)
    axis.set_aspect("equal")
    axis.set_xlim(lims)
    axis.set_ylim(lims)

    pyplot.xlabel(r"Reference $\Delta G$ [kcal/mol]")
    pyplot.ylabel(r"Estimated $\Delta G$ [kcal/mol]")

    pyplot.legend()

    pyplot.tight_layout()
    pyplot.savefig(output_path)


if __name__ == "__main__":
    main()
