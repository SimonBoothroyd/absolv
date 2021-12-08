import os.path
from glob import glob

import openmm.app
import openmm.unit
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator

import absolv
from absolv.models import (
    EquilibriumProtocol,
    NonEquilibriumProtocol,
    SimulationProtocol,
    State,
    SwitchingProtocol,
    System,
    TransferFreeEnergySchema,
)
from absolv.runners.equilibrium import EquilibriumRunner
from absolv.runners.nonequilibrium import NonEquilibriumRunner
from absolv.utilities.openmm import SystemGenerator, create_system_generator

STATE = State(
    temperature=298.15 * openmm.unit.kelvin, pressure=1.0 * openmm.unit.atmosphere
)

LAMBDA_ELECTROSTATICS_VACUUM = [1.0, 0.75, 0.5, 0.25, 0.0]
# fmt: off
LAMBDA_ELECTROSTATICS_SOLVENT = [
    1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
]

LAMBDA_STERICS_VACUUM = [1.0, 1.0, 1.0, 1.0, 1.0]
# fmt: off
LAMBDA_STERICS_SOLVENT = [
    1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50,
    0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00,
]


def setup_equilibrium(
    system: System, system_generator: SystemGenerator, directory: str
):

    schema = TransferFreeEnergySchema(
        system=system,
        state=STATE,
        # vacuum lambda states
        alchemical_protocol_a=EquilibriumProtocol(
            production_protocol=SimulationProtocol(
                n_steps_per_iteration=1000,
                n_iterations=2500
            ),
            lambda_sterics=LAMBDA_STERICS_VACUUM,
            lambda_electrostatics=LAMBDA_ELECTROSTATICS_VACUUM,
        ),
        # solvent lambda states
        alchemical_protocol_b=EquilibriumProtocol(
            production_protocol=SimulationProtocol(
                n_steps_per_iteration=1000,
                n_iterations=2500
            ),
            lambda_sterics=LAMBDA_STERICS_SOLVENT,
            lambda_electrostatics=LAMBDA_ELECTROSTATICS_SOLVENT,
        )
    )

    EquilibriumRunner.setup(schema, system_generator, directory)


def setup_non_equilibrium(
    system: System, system_generator: SystemGenerator, directory: str
):

    schema = TransferFreeEnergySchema(
        system=system,
        state=STATE,
        # vacuum
        alchemical_protocol_a=NonEquilibriumProtocol(
            switching_protocol=SwitchingProtocol(
                n_electrostatic_steps=60,
                n_steps_per_electrostatic_step=100,  # 12 ps
                n_steric_steps=0,
                n_steps_per_steric_step=0,
            )
        ),
        # solvent
        alchemical_protocol_b=NonEquilibriumProtocol(
            switching_protocol=SwitchingProtocol(
                n_electrostatic_steps=60,
                n_steps_per_electrostatic_step=100,  # 12 ps
                n_steric_steps=190,
                n_steps_per_steric_step=100,  # 38 ps
            )
        )
    )

    # Create the input files needed to run the simulations.
    NonEquilibriumRunner.setup(schema, system_generator, directory)


def main():

    root_directory = f"absolv-{absolv.__version__}"
    os.makedirs(root_directory)

    # Define the regression systems
    systems = {
        name: System(solutes={solute: 1}, solvent_a=None, solvent_b={"O": n_waters})
        # n_waters take from 'halx/relative-solvation-inputs' commit ec698ac
        for name, solute, n_waters in [
            ("methane", "C", 693),
            ("methanol", "CO", 703),
            ("ethane", "CC", 723),
            ("toluene", "CC1=CC=CC=C1", 870),
            ("neopentane", "CC(C)(C)C", 867),
            ("2-methylfuran", "CC1=CC=CO1", 831),
            ("2-methylindole", "CC1=CC2=CC=CC=C2N1", 943),
            ("2-cyclopentanylindole", "C1(CCCC1)C=1NC2=CC=CC=C2C1", 1122),
            ("7-cyclopentanylindole", "C1(CCCC1)C=1C=CC=C2C=CNC12", 1064),
        ]
    }

    # Define the 'system generator' to parameterize molecules using
    force_field = openmm.app.ForceField("amber/tip3p_standard.xml")
    force_field.registerTemplateGenerator(
        GAFFTemplateGenerator(
            # Use the charges provided in 'halx/relative-solvation-inputs' commit ec698ac
            molecules=[
                Molecule.from_file(input_path)
                for input_path in glob(os.path.join("reference-charges", "*.mol2"))
            ],
            forcefield="gaff-1.8"
        ).generator
    )
    system_generator = create_system_generator(
        force_field,
        solvent_a_nonbonded_method=openmm.app.NoCutoff,
        solvent_b_nonbonded_method=openmm.app.PME,
        switch_distance=0.9 * openmm.unit.nanometer,
        nonbonded_cutoff=1.0 * openmm.unit.nanometer,
        constraints=openmm.app.HBonds,
        rigid_water=True
    )

    for name, system in systems.items():

        setup_equilibrium(
            system, system_generator, os.path.join(root_directory, "eq", name)
        )
        setup_non_equilibrium(
            system, system_generator, os.path.join(root_directory, "neq", name)
        )


if __name__ == '__main__':
    main()
