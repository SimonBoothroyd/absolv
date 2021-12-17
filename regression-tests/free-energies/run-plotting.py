import json
import os.path
from collections import defaultdict
from glob import glob

import numpy
import pandas
import seaborn
from matplotlib import pyplot


def results_dict_to_pandas(version, results_dict):

    return [
        {"version": version, "method": method, "name": name, **result_dict}
        for method, method_results in results_dict.items()
        for name, result_dict in method_results.items()
    ]


def error_plot(x, y, yerr, hue, **kwargs):

    data = kwargs.pop("data")

    largest_error = data[yerr].max()

    minimum = data[y].min() - largest_error - 0.1
    maximum = data[y].max() + largest_error + 0.1

    p = data.pivot_table(index=x, columns=hue, values=y, aggfunc="mean", dropna=False)
    err = data.pivot_table(
        index=x, columns=hue, values=yerr, aggfunc="mean", dropna=False
    )

    p.plot(kind="bar", yerr=err, ax=pyplot.gca(), **kwargs)

    pyplot.gca().set_ylim([minimum, maximum])


def main():

    with open("reference-free-energies.json") as file:
        reference_data = json.load(file)

    pandas_data = results_dict_to_pandas("reference", {"GROMACS": reference_data})

    for path in glob(os.path.join("results", "absolv-*.json")):

        with open(path) as file:
            summary_dict = json.load(file)

        replica_values = defaultdict(list)

        for replica in (1, 2, 3):

            for result_dict in results_dict_to_pandas(
                summary_dict["version"], summary_dict["results"][f"{replica}"]
            ):

                key = (
                    result_dict["version"],
                    result_dict["method"],
                    result_dict["name"],
                )
                replica_values[key].append(result_dict["value"])

        for version, method, name in replica_values:

            if not any(v is None for v in replica_values[(version, method, name)]):
                mean = float(numpy.mean(replica_values[(version, method, name)]))
                std_error = float(numpy.std(replica_values[(version, method, name)]))
            else:
                mean = numpy.nan
                std_error = 0.0

            pandas_data.append(
                {
                    "version": version,
                    "method": method,
                    "name": name,
                    "value": mean,
                    "std_error": std_error,
                    "units": "kcal / mol",
                }
            )

    pandas_frame = pandas.DataFrame(pandas_data)

    methods = ["eq-indep", "eq-repex", "neq", "yank", "GROMACS"]
    palette = seaborn.color_palette(n_colors=len(methods))

    plot = seaborn.FacetGrid(
        pandas_frame,
        col="name",
        col_wrap=3,
        sharex=True,
        sharey=False,
        aspect=2.0,
        hue_order=methods,
    )
    plot.map_dataframe(
        error_plot,
        "version",
        "value",
        "std_error",
        "method",
        width=0.8,
        color=palette,
    )

    pyplot.subplots_adjust(right=0.90)
    pyplot.legend(loc="center left", bbox_to_anchor=(1, 1))

    pyplot.show()


if __name__ == "__main__":
    main()
