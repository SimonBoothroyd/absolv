<h1 align="center">ABsolute SOLVantion Free Energy Calculations</h1>

<p align="center">Absolute solvation free energy calculations using OpenMM</p>

<p align="center">
  <a href="https://github.com/SimonBoothroyd/absolv/actions?query=workflow%3Aci">
    <img alt="ci" src="https://github.com/SimonBoothroyd/absolv/actions/workflows/ci.yaml/badge.svg" />
  </a>
  <a href="https://codecov.io/gh/SimonBoothroyd/absolv/branch/main">
    <img alt="coverage" src="https://codecov.io/gh/SimonBoothroyd/absolv/branch/main/graph/badge.svg" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="license" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>

---

The `absolv` framework aims to offer a simple API for computing the change in free energy when transferring a solute
from one solvent to another, or to vacuum in the case of solvation free energy calculations.

It offers two routes to this end: standard equilibrium calculations and non-equilibrium switching type calculations,
where the latter will be the main focus of this framework.

***Warning**: This code is currently experimental and under active development. If you are using this it, please be
aware that it is not guaranteed to provide correct results, the documentation and testing is incomplete, and the
API can change without notice.*

## Installation

This package can be installed using `conda` (or `mamba`, a faster version of `conda`):

```shell
mamba install -c conda-forge femto
```

If you are running with MPI on an HPC cluster, you may need to instruct conda to use your local installation
depending on your setup

```shell
mamba install -c conda-forge femto "openmpi=4.1.5=*external*"
```

## Getting Started

To get started, see the [usage guide](https://simonboothroyd.github.io/absolv/latest/user-guide/overview/).
