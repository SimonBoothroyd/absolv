Non-Equilibrium Calculations
============================

To begin with we define a schema that will, for all intents and purposes, encode the entirety of the absolute free
energy calculation that we will be running.

In particular, we must specify:

i) the solutes and the two solvents that they will be transferred between
ii) the state, i.e. temperature and optionally pressure, to perform the calculation at
iii) and finally, the type of free energy calculation to perform as well as the protocol to follow while perform it

This can be compactly done by creating a new ``TransferFreeEnergySchema`` object:

.. code-block:: python

    from openmm import unit

    from absolv.models import (
        NonEquilibriumProtocol,
        SimulationProtocol,
        State,
        System,
        TransferFreeEnergySchema
    )

    schema = TransferFreeEnergySchema(
        system=System(solutes={"CCO": 1}, solvent_a={"O": 895}, solvent_b=None),
        # Define the state that the calculation will be performed at.
        state=State(temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere),
        # Define the alchemical pathway to transform the solute along in the first
        # and second (i.e. vacuum) solvent respectively.
        alchemical_protocol_a=NonEquilibriumProtocol(
            production_protocol=SimulationProtocol(
                n_steps_per_iteration=6250,
                n_iterations=160,
                timestep=2.0 * unit.femtoseconds
            ),
            switching_protocol=SwitchingProtocol(
                # Annihilate the electrostatic interactions over the first 12 ps
                n_electrostatic_steps=60,
                n_steps_per_electrostatic_step=100,
                # followed by decoupling the vdW interactions over the next 38 ps
                n_steric_steps=190,
                n_steps_per_steric_step=100,
            )
        ),
        alchemical_protocol_b=NonEquilibriumProtocol(
            production_protocol=SimulationProtocol(
                n_steps_per_iteration=6250,
                n_iterations=160,
                timestep=2.0 * unit.femtoseconds
            ),
            switching_protocol=SwitchingProtocol(
                n_electrostatic_steps=60,
                n_steps_per_electrostatic_step=100,
                n_steric_steps=0,
                n_steps_per_steric_step=0,
                timestep=2.0*unit.femtoseconds
            )
        )
    )

Here we have specified that we will be computing the solvation free energy of one ethanol molecule in
895 water molecules at a temperature and pressure of 298.15 K and 1 atm.

We have also specified that we will run an equilibrium simulation for 2 ns collecting a configuration snapshot every
6250 timesteps (12.5 ps), yielding 160 configurations, and for each of these configurations perform the non-equilibrium
switching simulations over the coarse of 50 ps.

We can then trivially set up,

.. code-block:: python

    from openff.toolkit.typing.engines.smirnoff import ForceField
    force_field = ForceField("openff-2.0.0.offxml")

    from absolv.runners.equilibrium import NonEquilibriumRunner
    NonEquilibriumRunner.setup(schema, force_field)

run,

.. code-block:: python

    NonEquilibriumRunner.run(schema, platform="CUDA")

and analyze the calculations

.. code-block:: python

    result = NonEquilibriumRunner.analyze(schema)
    print(result)

References
----------

.. bibliography:: non-equilibrium.bib
    :cited:
    :style: unsrt
