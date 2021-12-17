# Free Energy Regression Tests

This directory contains scripts to reproduce the absolute free energy calculations repored in

> Loeffler, Hannes H., et al. "Reproducibility of free energy calculations across different molecular simulation 
> software packages." Journal of chemical theory and computation 14.11 (2018): 5567-5582.

using the currently installed version of the `absolv` framework.

The reference free energies recorded in `reference-free-energies.json` were manually transcribed from the GROMACS 
column of Table 2.

To run the regression tests

1) Download the charges used in the study by running `download-charges.sh`
2) Create the input files by running `submit-setup.sh`
3) Submit all the required calculations to an LSF compute cluster using `submit-production.sh`
4) Summarise the results using `python run-summarise.py`
5) Plot the final data using `python run-plotting.py`
