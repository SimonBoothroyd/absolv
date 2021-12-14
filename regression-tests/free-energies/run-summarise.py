import json
import os.path
from collections import defaultdict
from glob import glob

import openmm.unit

import absolv
from absolv.models import TransferFreeEnergyResult


def main():

    root_directory = f"absolv-{absolv.__version__}"

    summary_dict = {
        "version": absolv.__version__,
        "results": defaultdict(lambda: defaultdict(dict)),
    }

    for replica in (1, 2, 3):

        for path in glob(os.path.join(f"{root_directory}-{replica}", "*", "*")):

            _, method, name = path.split(os.sep)

            result = TransferFreeEnergyResult.parse_file(
                os.path.join(path, "free-energies.json")
            )

            value, std_error = result.delta_g_from_a_to_b_with_units

            result_dict = {
                "value": value.value_in_unit(openmm.unit.kilocalorie_per_mole),
                "std_error": std_error.value_in_unit(openmm.unit.kilocalorie_per_mole),
                "units": "kcal / mol",
            }

            summary_dict["results"][replica][method][name] = result_dict

    with open(os.path.join("results", f"{root_directory}.json"), "w") as file:
        json.dump(summary_dict, file)


if __name__ == "__main__":
    main()
