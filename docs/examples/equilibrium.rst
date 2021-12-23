Equilibrium Calculations
========================

To begin with we define a schema that will, for all intents and purposes, encode the entirety of the absolute free
energy calculation that we will be running.

In particular, we must specify:

i) the solutes and the two solvents that they will be transferred between
ii) the state, i.e. temperature and optionally pressure, to perform the calculation at
iii) and finally, the type of free energy calculation to perform as well as the protocol to follow while perform it

This can be compactly done by creating a new ``TransferFreeEnergySchema`` object:

.. code-block:: python

    from openmm import unit

    from absolv.models import EquilibriumProtocol, State, System, TransferFreeEnergySchema

    schema = TransferFreeEnergySchema(
        # Define the solutes in the system. There may be multiple in the case of,
        # e.g., ion pairs like Na+ + Cl-. Here we use `None` to specify that the solute
        # will be transferred from vacuum into a water
        system=System(solutes={"CCO": 1}, solvent_a=None, solvent_b={"O": 895}),
        # Define the state that the calculation will be performed at.
        state=State(temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere),
        # Define the alchemical pathway to transform the solute along in vacuum ('solvent_a')
        # and water ('solvent_b')
        alchemical_protocol_a=EquilibriumProtocol(
            lambda_sterics=[1.0, 1.0, 1.0, 1.0, 1.0],
            lambda_electrostatics=[1.0, 0.75, 0.5, 0.25, 0.0],
            sampler="repex"
        ),
        alchemical_protocol_b=EquilibriumProtocol(
            lambda_sterics=[
                1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40,
                0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00,
            ],
            lambda_electrostatics=[
                1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
            ],
            sampler="repex"
        ),
    )

Here we have specified that we will be computing the solvation free energy of one ethanol molecule in
895 water molecules at a temperature and pressure of 298.15 K and 1 atm.

We have also specified that we will be performing a standard 'equilibrium' free energy calculation according
to the paper described by Mobley et al :cite:`mobley2018escaping`.

We can then trivially set up,

.. code-block:: python

    from openff.toolkit.typing.engines.smirnoff import ForceField
    force_field = ForceField("openff-2.0.0.offxml")

    from absolv.runners.equilibrium import EquilibriumRunner
    EquilibriumRunner.setup(schema, force_field)

run,

.. code-block:: python

    EquilibriumRunner.run(schema, platform="CUDA")

and analyze the calculations

.. code-block:: python

    result = EquilibriumRunner.analyze(schema)
    print(result)

References
----------

.. bibliography:: equilibrium.bib
    :cited:
    :style: unsrt
