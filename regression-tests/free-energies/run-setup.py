import os.path
from glob import glob
from typing import Literal

import click
import openmm.app
import openmm.unit
import yaml
from openff.toolkit.topology import Molecule
from openff.utilities import temporary_cd
from openmm import unit
from openmmforcefields.generators import GAFFTemplateGenerator

import absolv
from absolv.factories.coordinate import PACKMOLCoordinateFactory
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

CUTOFF = 1.0 * openmm.unit.nanometer

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


def default_equilibrium_schema(
    system: System, sampler: Literal["independent", "repex"]
) -> TransferFreeEnergySchema:

    return TransferFreeEnergySchema(
        system=system,
        state=STATE,
        # vacuum lambda states
        alchemical_protocol_a=EquilibriumProtocol(
            production_protocol=SimulationProtocol(
                n_steps_per_iteration=500,
                n_iterations=3000
            ),
            lambda_sterics=LAMBDA_STERICS_VACUUM,
            lambda_electrostatics=LAMBDA_ELECTROSTATICS_VACUUM,
            sampler=sampler
        ),
        # solvent lambda states
        alchemical_protocol_b=EquilibriumProtocol(
            production_protocol=SimulationProtocol(
                n_steps_per_iteration=500,
                n_iterations=3000
            ),
            lambda_sterics=LAMBDA_STERICS_SOLVENT,
            lambda_electrostatics=LAMBDA_ELECTROSTATICS_SOLVENT,
            sampler=sampler
        )
    )


def setup_equilibrium(
    system: System,
    system_generator: SystemGenerator,
    directory: str,
    sampler: Literal["independent", "repex"]
):

    schema = default_equilibrium_schema(system, sampler)
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


def setup_yank(
    system: System, system_generator: SystemGenerator, directory: str
):

    schema = default_equilibrium_schema(system, "repex")

    topology_a, coordinates_a = PACKMOLCoordinateFactory.generate(
        [
            *system.solutes.items(),
            *({} if system.solvent_a is None else system.solvent_a).items()
        ]
    )
    topology_a.box_vectors = None if system.solvent_a is None else topology_a.box_vectors
    system_a = system_generator(topology_a, coordinates_a, "solvent-a")

    topology_b, coordinates_b = PACKMOLCoordinateFactory.generate(
        [
            *system.solutes.items(),
            *({} if system.solvent_b is None else system.solvent_b).items()
        ]
    )
    topology_b.box_vectors = None if system.solvent_b is None else topology_b.box_vectors
    system_b = system_generator(topology_b, coordinates_b, "solvent-b")

    input_dict = dict(
        options=dict(
            output_dir=".",
            temperature=f"{schema.state.temperature} * kelvin",
            pressure=f"{schema.state.pressure} * atmosphere",
            minimize=True,
            number_of_equilibration_iterations=150,
            equilibration_timestep="2.0 * femtosecond",
            default_number_of_iterations=3000,
            default_nsteps_per_iteration=500,
            default_timestep="2.0 * femtosecond",
            annihilate_electrostatics=True,
            annihilate_sterics=False,
            # Try and get Yank to setup the system as close to absolv does as possible.
            # Namely, don't use a direct-space soft-core for electrostatics
            alchemical_pme_treatment="exact",
            # Don't try to expand the cut-off to a large value by re-weighting
            anisotropic_dispersion_cutoff=(
                f"{CUTOFF.value_in_unit(unit.nanometers)} * nanometers"
            ),
        ),
        systems=dict(
            default=dict(
                phase1_path=["solvent-a.xml", "solvent-a.pdb"],
                phase2_path=["solvent-b.xml", "solvent-b.pdb"],
                solvent_dsl="chainid 1"
            )
        ),
        protocols=dict(
            default=dict(
                solvent1={
                    "alchemical_path": {
                        "lambda_electrostatics": schema.alchemical_protocol_a.lambda_electrostatics,
                        "lambda_sterics": schema.alchemical_protocol_a.lambda_sterics,
                    }
                },
                solvent2={
                    "alchemical_path": {
                        "lambda_electrostatics": schema.alchemical_protocol_b.lambda_electrostatics,
                        "lambda_sterics": schema.alchemical_protocol_b.lambda_sterics
                    }
                },
            )
        ),
        experiments=dict(
            system="default",
            protocol="default"
        )
    )

    os.makedirs(directory)

    with temporary_cd(directory):

        with open("solvent-a.xml", "w") as file:
            file.write(openmm.XmlSerializer.serialize(system_a))
        with open("solvent-b.xml", "w") as file:
            file.write(openmm.XmlSerializer.serialize(system_b))

        topology_a.to_file("solvent-a.pdb", coordinates_a)
        topology_b.to_file("solvent-b.pdb", coordinates_b)

        with open("yank.yaml", "w") as file:
            yaml.dump(input_dict, file, sort_keys=False)

        with open("schema.json", "w") as file:
            file.write(schema.json(indent=4))


@click.argument("replica", type=click.INT)
@click.command()
def main(replica):

    print(f"PACKMOL seed={os.environ['ABSOLV_PACKMOL_SEED']}")

    root_directory = f"absolv-{absolv.__version__}-{replica}"
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
        nonbonded_cutoff=CUTOFF,
        constraints=openmm.app.HBonds,
        rigid_water=True
    )

    for name, system in systems.items():

        setup_equilibrium(
            system,
            system_generator,
            os.path.join(root_directory, "eq-indep", name),
            "independent"
        )
        setup_equilibrium(
            system,
            system_generator,
            os.path.join(root_directory, "eq-repex", name),
            "repex"
        )

        setup_non_equilibrium(
            system, system_generator, os.path.join(root_directory, "neq", name)
        )
        setup_yank(
            system, system_generator, os.path.join(root_directory, "yank", name)
        )


if __name__ == '__main__':
    main()
