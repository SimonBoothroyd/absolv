import logging
import pathlib
import tempfile
import urllib.request
import datetime

import click
import femto.md.config
import femto.md.system
import openff.toolkit
import openff.units
import openmm.app
import openmm.unit
import openmmforcefields.generators
import parmed
from rdkit import Chem

import absolv.config
import absolv.utils.openmm
import absolv.runner

DEFAULT_TEMPERATURE = 298.15 * openmm.unit.kelvin
DEFAULT_PRESSURE = 1.0 * openmm.unit.atmosphere

DEFAULT_SYSTEMS = {
    name: absolv.config.System(
        solutes={solute: 1}, solvent_a=None, solvent_b={"O": n_waters}
    )
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
DEFAULT_SOLUTES = [*DEFAULT_SYSTEMS]

DEFAULT_METHODS = ["eq", "neq"]
DEFAULT_REPLICAS = ["0", "1", "2"]

REFERENCE_COMMIT = "ec698ac"
REFERENCE_URL = (
    f"https://raw.githubusercontent.com/"
    f"halx/relative-solvation-inputs/{REFERENCE_COMMIT}/FESetup/setup"
)


def default_config_neq(system: absolv.config.System) -> absolv.config.Config:
    return absolv.config.Config(
        system=system,
        temperature=DEFAULT_TEMPERATURE,
        pressure=DEFAULT_PRESSURE,
        alchemical_protocol_a=absolv.config.NonEquilibriumProtocol(
            switching_protocol=absolv.config.SwitchingProtocol(
                n_electrostatic_steps=60,
                n_steps_per_electrostatic_step=100,  # 12 ps
                n_steric_steps=0,
                n_steps_per_steric_step=0,
            )
        ),
        alchemical_protocol_b=absolv.config.NonEquilibriumProtocol(
            switching_protocol=absolv.config.SwitchingProtocol(
                n_electrostatic_steps=60,
                n_steps_per_electrostatic_step=100,  # 12 ps
                n_steric_steps=190,
                n_steps_per_steric_step=100,  # 38 ps
            )
        ),
    )


def default_config_eq(system: absolv.config.System) -> absolv.config.Config:
    return absolv.config.Config(
        system=system,
        temperature=DEFAULT_TEMPERATURE,
        pressure=DEFAULT_PRESSURE,
        alchemical_protocol_a=absolv.config.EquilibriumProtocol(
            production_protocol=absolv.config.HREMDProtocol(
                n_steps_per_cycle=500,
                n_cycles=2000,
                integrator=femto.md.config.LangevinIntegrator(
                    timestep=1.0 * openmm.unit.femtosecond
                ),
            ),
            lambda_sterics=absolv.config.DEFAULT_LAMBDA_STERICS_VACUUM,
            lambda_electrostatics=absolv.config.DEFAULT_LAMBDA_ELECTROSTATICS_VACUUM,
        ),
        alchemical_protocol_b=absolv.config.EquilibriumProtocol(
            production_protocol=absolv.config.HREMDProtocol(
                n_steps_per_cycle=500,
                n_cycles=1000,
                integrator=femto.md.config.LangevinIntegrator(
                    timestep=4.0 * openmm.unit.femtosecond
                ),
            ),
            lambda_sterics=absolv.config.DEFAULT_LAMBDA_STERICS_SOLVENT,
            lambda_electrostatics=absolv.config.DEFAULT_LAMBDA_ELECTROSTATICS_SOLVENT,
        ),
    )


def _download_ref_mols():
    # Use the charges provided in 'halx/relative-solvation-inputs' for consistency
    ref_mols = []

    for solute in DEFAULT_SOLUTES:
        solute_url = f"{REFERENCE_URL}/{solute}/gaff.mol2"

        with tempfile.TemporaryDirectory() as tmp_dir:
            path_mol2 = str(pathlib.Path(tmp_dir, f"{solute}.mol2"))
            path_mol = str(pathlib.Path(tmp_dir, f"{solute}.mol"))

            urllib.request.urlretrieve(solute_url, path_mol2)

            mol_pmd = parmed.load_file(path_mol2, structure=True)

            for atom in mol_pmd.atoms:
                atom.type = atom.element_name

            mol_pmd.save(path_mol2, overwrite=True)

            mol_rdkit: Chem.Mol = Chem.MolFromMol2File(path_mol2, removeHs=False)
            Chem.MolToMolFile(mol_rdkit, path_mol)

            charges = [atom.charge for atom in mol_pmd.atoms] * openff.units.unit.e

            mol_openff = openff.toolkit.Molecule.from_file(path_mol)
            mol_openff.partial_charges = charges

            ref_mols.append(mol_openff)

    return ref_mols


def default_system_generator() -> absolv.utils.openmm.SystemGenerator:
    ref_mols = _download_ref_mols()

    force_field = openmm.app.ForceField("amber/tip3p_standard.xml")
    force_field.registerTemplateGenerator(
        openmmforcefields.generators.GAFFTemplateGenerator(
            molecules=ref_mols, forcefield="gaff-1.8"
        ).generator
    )

    system_generator = absolv.utils.openmm.create_system_generator(
        force_field,
        solvent_a_nonbonded_method=openmm.app.NoCutoff,
        solvent_b_nonbonded_method=openmm.app.PME,
        switch_distance=0.9 * openmm.unit.nanometer,
        nonbonded_cutoff=1.0 * openmm.unit.nanometer,
        constraints=openmm.app.HBonds,
        rigid_water=True,
    )
    return system_generator


def run_replica(
    system: absolv.config.System,
    system_generator: absolv.utils.openmm.SystemGenerator,
    method: str,
    output_dir: pathlib.Path,
):
    output_dir.mkdir(parents=True, exist_ok=False)

    config_generator = default_config_neq if method == "neq" else default_config_eq
    config = config_generator(system)

    prepared_system_a, prepared_system_b = absolv.runner.setup(config, system_generator)

    femto.md.system.apply_hmr(
        prepared_system_a.system,
        parmed.openmm.load_topology(prepared_system_a.topology.to_openmm()),
    )
    femto.md.system.apply_hmr(
        prepared_system_b.system,
        parmed.openmm.load_topology(prepared_system_a.topology.to_openmm()),
    )

    run_fn = absolv.runner.run_neq if method == "neq" else absolv.runner.run_eq
    result = run_fn(config, prepared_system_a, prepared_system_b, "CUDA")

    (output_dir / "result.json").write_text(result.model_dump_json(indent=2))


@click.command()
@click.option(
    "--solute",
    "solutes",
    multiple=True,
    type=click.Choice(DEFAULT_SOLUTES),
    default=DEFAULT_SOLUTES,
)
@click.option(
    "--method",
    "methods",
    multiple=True,
    type=click.Choice(DEFAULT_METHODS),
    default=DEFAULT_METHODS,
)
@click.option(
    "--replica", "replicas", multiple=True, type=str, default=DEFAULT_REPLICAS
)
def main(solutes: list[str], methods: list[str], replicas: list[str]):
    logging.basicConfig(level=logging.INFO)

    output_name = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    output_dir = pathlib.Path("results", output_name)

    system_generator = default_system_generator()

    for solute in solutes:
        for method in methods:
            for replica in replicas:
                logging.info(f"running {method} {solute} {replica}")

                replica_dir = output_dir / f"{method}-{solute}-{replica}"
                run_replica(
                    DEFAULT_SYSTEMS[solute], system_generator, method, replica_dir
                )


if __name__ == "__main__":
    main()
