# Free Energy Regression Tests

This directory contains scripts to reproduce the absolute free energy calculations
reported in

> Loeffler, Hannes H., et al. "Reproducibility of free energy calculations across different molecular simulation
> software packages." Journal of chemical theory and computation 14.11 (2018): 5567-5582.

using the currently installed version of the `absolv` framework.

The reference free energies recorded in `results/reference.json` were manually
transcribed from the GROMACS column of Table 2.

To run the default set of regression tests:

```shell
python run.py
```

A subset of the tests can be run using the `--solute`, `--method` and `--replica` flags:

```shell
python run.py --solute methanol \
              --solute toluene  \
              --method neq      \
              --method eq       \
              --replica 0       \
              --replica 1
```

The results will be written to `results/{timestamp}/{method}-{solute}-{replica}/results.json`.
