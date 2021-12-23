import copy

import numpy
import openmm
import pytest
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import unit

from absolv.factories.alchemical import (
    OpenMMAlchemicalFactory,
    lj_potential,
    lorentz_berthelot,
    soft_core_lj_potential,
)
from absolv.factories.coordinate import PACKMOLCoordinateFactory
from absolv.tests import is_close
from absolv.utilities.openmm import disable_long_range_corrections, evaluate_energy
from absolv.utilities.topology import topology_to_atom_indices


def _convert_nonbonded_to_custom(system: openmm.System) -> openmm.System:
    """Moves the vdW interactions from a ``NonbondedForce`` to a custom non-bonded and
    bond force.
    """

    system = copy.deepcopy(system)

    [original_force] = [
        force
        for force in system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ]

    custom_n_force = openmm.CustomNonbondedForce(lj_potential() + lorentz_berthelot())
    custom_n_force.setNonbondedMethod(
        original_force.getNonbondedMethod()
        if int(original_force.getNonbondedMethod()) not in {3, 4, 5}
        else openmm.CustomNonbondedForce.CutoffPeriodic
    )
    custom_n_force.setCutoffDistance(original_force.getCutoffDistance())
    custom_n_force.setSwitchingDistance(original_force.getSwitchingDistance())
    custom_n_force.setUseSwitchingFunction(original_force.getUseSwitchingFunction())
    custom_n_force.setUseLongRangeCorrection(
        original_force.getUseDispersionCorrection()
    )
    custom_n_force.addPerParticleParameter("sigma")
    custom_n_force.addPerParticleParameter("epsilon")

    for index in range(original_force.getNumParticles()):

        charge, sigma, epsilon = original_force.getParticleParameters(index)
        custom_n_force.addParticle([sigma, epsilon])
        original_force.setParticleParameters(index, charge, sigma, epsilon * 0)

    custom_b_force = openmm.CustomBondForce(lj_potential())
    custom_b_force.addPerBondParameter("sigma")
    custom_b_force.addPerBondParameter("epsilon")

    for index in range(original_force.getNumExceptions()):

        (
            index_a,
            index_b,
            charge,
            sigma,
            epsilon,
        ) = original_force.getExceptionParameters(index)

        custom_n_force.addExclusion(index_a, index_b)
        custom_b_force.addBond(index_a, index_b, [sigma, epsilon])

        original_force.setExceptionParameters(
            index, index_a, index_b, charge, sigma, epsilon * 0
        )

    system.addForce(custom_n_force)
    system.addForce(custom_b_force)

    return system


@pytest.mark.parametrize(
    "force_field",
    [
        ForceField("openff-2.0.0.offxml"),
    ],
)
@pytest.mark.parametrize(
    "smiles",
    [
        "O",
        "CO",
        "CCO",
        "NC(=O)N",
        "CC1C(C)C(C)C1C",
        "OC[C@@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    ],
)
def test_vacuum_energies(force_field, smiles):
    """Checks that an fully interacting alchemically modified system yields the same
    energy in vacuum as an un-modified system"""

    topology, coordinates = PACKMOLCoordinateFactory.generate([(smiles, 1)])
    topology.box_vectors = None

    system_chemical = force_field.create_openmm_system(topology)

    system_alchemical = OpenMMAlchemicalFactory.generate(
        copy.deepcopy(system_chemical), topology_to_atom_indices(topology), []
    )

    energy_chemical = evaluate_energy(system_chemical, coordinates, None, None, None)
    energy_alchemical = evaluate_energy(system_alchemical, coordinates, None, 1.0, 1.0)

    assert is_close(energy_alchemical, energy_chemical)


@pytest.mark.parametrize(
    "force_field",
    [
        ForceField("openff-2.0.0.offxml"),
    ],
)
@pytest.mark.parametrize(
    "solutes, solvents",
    [
        ([("[Na+]", 1), ("[Cl-]", 1)], [("O", 216)]),
        ([("CCO", 1)], [("O", 216)]),
        ([("OC[C@@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O", 1)], [("CO", 216)]),
    ],
)
def test_solution_energies(force_field, solutes, solvents):
    """Checks that an fully interacting alchemically modified system yields the same
    energy for a solution as an un-modified system"""

    topology, coordinates = PACKMOLCoordinateFactory.generate(solutes + solvents)
    topology.box_vectors = None if len(solvents) == 0 else topology.box_vectors

    atom_indices = topology_to_atom_indices(topology)
    n_solutes = sum(count for _, count in solutes)

    system_chemical = force_field.create_openmm_system(topology)
    # disable LRC as the custom force does not analytically compute these leading
    # to slight differences in energies in solution.
    disable_long_range_corrections(system_chemical)

    system_alchemical = OpenMMAlchemicalFactory.generate(
        system_chemical, atom_indices[:n_solutes], atom_indices[n_solutes:]
    )

    energy_chemical = evaluate_energy(system_chemical, coordinates, None, None, None)
    energy_alchemical = evaluate_energy(system_alchemical, coordinates, None, 1.0, 1.0)

    assert is_close(energy_alchemical, energy_chemical)


def test_lj_vs_custom_lj_energy():
    """Make sure that a system that contains the built-in LJ function and a system
    that contains an LJ function stored in a custom force yield the same energies after
    alchemical transformation."""

    force_field = ForceField("openff-2.0.0.offxml")

    for solvent_index, components in [
        ("solvent-a", [("CC(C)(C)C", 1)]),
        ("solvent-b", [("O", 256), ("CC(C)(C)C", 1)]),
    ]:

        topology, coordinates = PACKMOLCoordinateFactory.generate(components)
        topology.box_vectors = (
            None if solvent_index == "solvent-a" else topology.box_vectors
        )

        atom_indices = topology_to_atom_indices(topology)

        alchemical_indices = atom_indices[:1]
        persistent_indices = atom_indices[1:]

        system_original = force_field.create_openmm_system(topology)
        # disable LRC as the custom force does not analytically compute these leading
        # to slight differences in energies in solution.
        disable_long_range_corrections(system_original)

        system_custom = _convert_nonbonded_to_custom(system_original)

        energy_original = evaluate_energy(
            system_original, coordinates, topology.box_vectors
        ).value_in_unit(unit.kilocalorie_per_mole)
        energy_custom = evaluate_energy(
            system_custom, coordinates, topology.box_vectors
        ).value_in_unit(unit.kilocalorie_per_mole)

        print("", flush=True)
        print(
            f"{solvent_index} REF",
            f"Eorig={energy_original:.6f}",
            f"Ecust={energy_custom:.6f}",
            f"ΔE={energy_custom - energy_original:.6f}",
        )
        print("", flush=True)

        assert numpy.isclose(energy_custom, energy_original)

        alc_system_original = OpenMMAlchemicalFactory.generate(
            system_original,
            alchemical_indices,
            persistent_indices,
        )
        alc_system_custom = OpenMMAlchemicalFactory.generate(
            system_custom,
            alchemical_indices,
            persistent_indices,
            custom_alchemical_potential=soft_core_lj_potential() + lorentz_berthelot(),
        )

        for lambda_electrostatics, lambda_sterics in [
            (1.0, 1.0),
            (0.5, 1.0),
            (0.0, 1.0),
            (0.0, 0.5),
            (0.0, 0.0),
        ]:

            energy_original = evaluate_energy(
                alc_system_original,
                coordinates,
                topology.box_vectors,
                lambda_electrostatics=lambda_electrostatics,
                lambda_sterics=lambda_sterics,
            ).value_in_unit(unit.kilocalorie_per_mole)
            energy_custom = evaluate_energy(
                alc_system_custom,
                coordinates,
                topology.box_vectors,
                lambda_electrostatics=lambda_electrostatics,
                lambda_sterics=lambda_sterics,
            ).value_in_unit(unit.kilocalorie_per_mole)

            print(
                f"λE={lambda_electrostatics:.2f}",
                f"λv={lambda_sterics:.2f}",
                f"Eorig={energy_original:.6f}",
                f"Ecust={energy_custom:.6f}",
                f"ΔE={energy_custom - energy_original:.6f}",
            )

            assert numpy.isclose(energy_custom, energy_original)


def two_particle_built_in_system(cutoff, method) -> openmm.System:

    system = openmm.System()

    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.19, 1.0)
    system.addForce(bond)

    force = openmm.NonbondedForce()
    force.setNonbondedMethod(method)
    force.setCutoffDistance(cutoff)
    force.setUseDispersionCorrection(False)
    force.setUseSwitchingFunction(False)

    for i in range(2):
        system.addParticle(1.0)
        force.addParticle(0.0, 0.005, 1.0)

    system.addForce(force)
    return system


def two_particle_custom_system(cutoff, method) -> openmm.System:

    # edge of the periodic box.
    system = openmm.System()

    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.19, 1.0)
    system.addForce(bond)

    force = openmm.CustomNonbondedForce("4.0*((0.005/r)^12 - (0.005/r)^6)")
    force.setNonbondedMethod(method)
    force.setCutoffDistance(cutoff)

    for i in range(2):
        system.addParticle(1.0)
        force.addParticle([])

    system.addForce(force)
    system.addForce(openmm.CustomBondForce("1/r"))

    return system


@pytest.mark.parametrize(
    "system_func", [two_particle_built_in_system, two_particle_custom_system]
)
@pytest.mark.parametrize(
    "method, expected_energy",
    [
        (openmm.CustomNonbondedForce.CutoffNonPeriodic, 0.0),
        (openmm.CustomNonbondedForce.CutoffPeriodic, 0.0),
        (
            openmm.CustomNonbondedForce.NoCutoff,
            4.0 * ((0.005 / 0.19) ** 12 - (0.005 / 0.19) ** 6),
        ),
    ],
)
def test_periodic_interactions(system_func, method, expected_energy):

    # Construct a contrived system with two atoms of the same molecule located at the
    # edges of the periodic box.
    system = system_func(0.975 * unit.angstrom, method)

    coordinates = numpy.array([[-0.95, 0.0, 0.0], [0.95, 0.0, 0.0]]) * unit.angstrom
    box_vectors = (numpy.eye(3) * 2.0) * unit.angstrom

    alchemical_system = OpenMMAlchemicalFactory.generate(system, [{0, 1}], [])
    alchemical_energy = evaluate_energy(
        alchemical_system, coordinates, box_vectors
    ).value_in_unit(unit.kilojoule_per_mole)

    assert numpy.isclose(alchemical_energy, expected_energy)
