# ABsolute SOLVantion Free Energy Calculations

[![Test Status](https://github.com/simonboothroyd/absolv/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/simonboothroyd/absolv/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/simonboothroyd/absolv/branch/main/graph/badge.svg)](https://codecov.io/gh/simonboothroyd/absolv/branch/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The `absolv` framework aims to offer a simple API for computing the change in free energy when transferring a solute 
from one solvent to another, or to vacuum in the case of solvation free energy calculations.

It offers two routes to this end: standard equilibrium calculations and non-equilibrium switching type calculations, 
where the latter will be the main focus of this framework.

***Warning**: This code is currently experimental and under active development. If you are using this it, please be 
aware that it is not guaranteed to provide correct results, the documentation and testing is incomplete, and the
API can change without notice.*

## Getting Started

To start using this framework we recommend looking over [the documentation](https://simonboothroyd.github.io/absolv/),
especially the [equilibrium](https://simonboothroyd.github.io/absolv/examples/equilibrium.html) and 
[non-equilibrium](https://simonboothroyd.github.io/absolv/examples/non-equilibrium.html) free energy examples.

## License

The main package is release under the [MIT license](LICENSE). 

## Copyright

Copyright (c) 2021, Simon Boothroyd
