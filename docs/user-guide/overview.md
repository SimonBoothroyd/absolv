# Overview

To begin with, we define a configuration that encodes the entirety of the absolute free
energy calculation.

This includes specifying the solutes and the two solvents that they will be
transferred between:

```python
import absolv.config

system=absolv.config.System(
    solutes={"CCO": 1}, solvent_a=None, solvent_b={"O": 895}
)
```

Each key should be a SMILES representation of a molecule, and the value should be the
number of copies of that molecule to include in the system. There may be multiple in the
case of, e.g., ion pairs like `Na+` and `Cl-`. `None` may be used to specify vacuum,
e.g., in the above case that the solute will be transferred from vacuum into bulk water.

The temperature and pressure that the calculation will be performed at must also be
specified:

```python
import openmm.unit

temperature=298.15 * openmm.unit.kelvin
pressure=1.0 * openmm.unit.atmosphere
```

Finally, the alchemical pathway to transform the solute along in each solvent must be
specified. This can either be a more traditional 'equilibrium' pathway, or a
'non-equilibrium' pathway:

=== "Equilibrium"

    ```python
    import absolv.config

    alchemical_protocol_a=absolv.config.EquilibriumProtocol(
        lambda_sterics=[1.0, 1.0, 1.0, 1.0, 1.0],
        lambda_electrostatics=[1.0, 0.75, 0.5, 0.25, 0.0]
    )
    alchemical_protocol_b=absolv.config.EquilibriumProtocol(
        lambda_sterics=[
            1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40,
            0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00,
        ],
        lambda_electrostatics=[
            1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        ]
    )
    ```

    Here the default lambda schedule from FreeSolv is used. A lambda of `1.0` indicates
    a fully interacting system, and a lambda of `0.0` indicates a fully decoupled
    system.

=== "Non-equilibrium"

    ```python
    import femto.md.config
    import openmm.unit

    import absolv.config

    integrator = femto.md.config.LangevinIntegrator(
        timestep=2.0 * openmm.unit.femtoseconds
    )

    alchemical_protocol_a=absolv.config.NonEquilibriumProtocol(
        # the protocol to use for the production run at each end state
        production_protocol=absolv.config.SimulationProtocol(
            integrator=integrator, n_steps=6250 * 160
        ),
        # the interval with which to store frames that NEQ switching
        # simulations will be launched from
        production_report_interval=6250,
        # define how the NEQ switching will be performed
        switching_protocol=absolv.config.SwitchingProtocol(
            n_electrostatic_steps=60,
            n_steps_per_electrostatic_step=100,
            # intra-molecular vdW interactions are not decoupled by default, so we
            # don't need to do any vdW decoupling in vacuum when there's only one solvent
            n_steric_steps=0,
            n_steps_per_steric_step=0
        )
    )
    alchemical_protocol_b=absolv.config.NonEquilibriumProtocol(
        production_protocol=absolv.config.SimulationProtocol(
            integrator=integrator, n_steps=6250 * 160,
        ),
        switching_protocol=absolv.config.SwitchingProtocol(
            # Annihilate the electrostatic interactions over the first 12 ps
            n_electrostatic_steps=60,
            n_steps_per_electrostatic_step=100,
            # followed by decoupling the vdW interactions over the next 38 ps
            n_steric_steps=190,
            n_steps_per_steric_step=100,
        )
    )
    ```

These individual components are then combined into a single configuration object:

```python
import absolv.config

config = absolv.config.Config(
    temperature=temperature,
    pressure=pressure,
    alchemical_protocol_a=alchemical_protocol_a,
    alchemical_protocol_b=alchemical_protocol_b,
)
```

which can be used to trivially set up

```python
import openff.toolkit
force_field = openff.toolkit.ForceField("openff-2.1.0.offxml")

import absolv.runner
prepared_system_a, prepared_system_b = absolv.runner.setup(system, config, force_field)
```

and run the calculation:

=== "Equilibrium"

    ```python
    result = absolv.runner.run_eq(
        config, prepared_system_a, prepared_system_b, "CUDA"
    )
    print(result)
    ```

=== "Non-equilibrium"

    ```python
    result = absolv.runner.run_neq(
        config, prepared_system_a, prepared_system_b, "CUDA"
    )
    print(result)
    ```

where the result will be a [Result][absolv.config.Result] object.
