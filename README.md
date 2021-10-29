ABsolute SOLVantion Free Energy Calculations
============================================
[![Test Status](https://github.com/simonboothroyd/absolv/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/simonboothroyd/absolv/actions/workflows/ci.yaml)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/SimonBoothroyd/absolv.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/SimonBoothroyd/absolv/context:python)
[![codecov](https://codecov.io/gh/simonboothroyd/absolv/branch/main/graph/badge.svg)](https://codecov.io/gh/simonboothroyd/absolv/branch/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The `absolv` framework aims to offer a simple API for computing the change in free energy when transferring a solute 
from one solvent to another, or to vacuum in the case of solvation free energy calculations.

It offers two routes to this end: standard equilibrium calculations and non-equilibrium switching type calculations, 
where the latter is the main focus of this framework.

***Warning**: This code is currently experimental and under active development. If you are using this it, please be 
aware that it is not guaranteed to provide correct results, the documentation and testing is incomplete, and the
API can change without notice.*

## Installation

The core dependencies can be installed using the [`conda`](https://docs.conda.io/en/latest/miniconda.html) 
package manager:

```shell
conda env create --name absolv --file devtools/conda-envs/test-env.yaml
python setup.py develop
```

## Getting Started

To begin with we define a schema that will, for all intents and purposes, encode the entirety of the absolute free 
energy calculation that we will be running.

In particular, we must specify:

i) the solutes and the two solvents that they will be transferred between
ii) the state, i.e. temperature and optionally pressure, to perform the calculation at
iii) and finally, the type of free energy calculation to perform as well as the protocol to follow while perform it

This can be compactly done by creating a new `TransferFreeEnergySchema` object:

```python
from openmm import unit

from absolv.models import EquilibriumProtocol, State, System, TransferFreeEnergySchema

schema = TransferFreeEnergySchema(
    # Define the solutes in the system. There may be multiple in the case of,
    # e.g., ion pairs like Na+ + Cl-. Here we use `None` to specify that the solute 
    # will be transferred into a vacuum.
    system=System(solutes={"CCO": 1}, solvent_a={"O": 895}, solvent_b=None),
    # Define the state that the calculation will be performed at.
    state=State(temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere),
    # Define the alchemical pathway to transform the solute along in the first
    # and second (i.e. vacuum) solvent respectively.
    alchemical_protocol_a=EquilibriumProtocol(
        lambda_sterics=[  # fmt: off
            1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40,
            0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00,
        ],
        lambda_electrostatics=[  # fmt: off
            1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        ],
    ),
    alchemical_protocol_b=EquilibriumProtocol(
        lambda_sterics=[1.0, 1.0, 1.0, 1.0, 1.0],
        lambda_electrostatics=[1.0, 0.75, 0.5, 0.25, 0.0],
    ),
)
```

Here we have specified that we will be computing the solvation free energy of one ethanol molecule in
895 water molecules at a temperature and pressure of 298.15 K and 1 atm.

We have also specified that we will be performing a standard 'equilibrium' free energy calculation according
to the paper described by Mobley et al [1].

We can then trivially set up,

```python
from openff.toolkit.typing.engines.smirnoff import ForceField
force_field = ForceField("openff-2.0.0.offxml")

from absolv.runners.equilibrium import EquilibriumRunner
EquilibriumRunner.setup(schema, force_field)
```

run,

```python
EquilibriumRunner.run(schema, platform="CUDA")
```

and analyze the calculations

```python
free_energies = EquilibriumRunner.analyze(schema)
print(free_energies)
```

## Theory

...

### Equilibrium Calculations

...

### Non-equilibrium Calculations

...

### Alchemical Transformations

...

#### Electrostatics

...

#### LJ Soft Core

...

#### Custom vdW Forms

...

## References

[1] Mobley, David L., et al. "Escaping atom types in force fields using direct chemical perception." [Journal of 
    chemical theory and computation 14.11 (2018): 6076-6092.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6245550/)

## License

The main package is release under the [MIT license](LICENSE). 

## Copyright

Copyright (c) 2021, Simon Boothroyd
