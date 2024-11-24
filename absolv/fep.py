"""Prepare OpenMM systems for FEP calculations."""

import copy
import itertools

import numpy
import openmm
import openmm.unit

_KJ_MOL = openmm.unit.kilojoule_per_mole
_E = openmm.unit.elementary_charge

LAMBDA_STERICS = "lambda_sterics"
LAMBDA_ELECTROSTATICS = "lambda_electrostatics"

LORENTZ_BERTHELOT = "sigma = 0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2);"

LJ_POTENTIAL = "4.0*epsilon*x*(x-1.0); x = (sigma/r)^6;"
LJ_POTENTIAL_BEUTLER = (
    f"{LAMBDA_STERICS}*4.0*epsilon*x*(x-1.0);"
    "x = (sigma/reff_sterics)^6;"
    f"reff_sterics = sigma*(0.5*(1.0-{LAMBDA_STERICS}) + (r/sigma)^6)^(1/6);"
)


def _find_nonbonded_forces(
    system: openmm.System,
) -> tuple[
    openmm.NonbondedForce | None,
    openmm.CustomNonbondedForce | None,
    openmm.CustomBondForce | None,
]:
    """Attempts to find the forces that describe the non-bonded (both vdW and
    electrostatic) interactions in the system.

    Args:
        system: The OpenMM system containing the forces.
    """

    normal_nonbonded_forces = []
    custom_nonbonded_forces = []
    # We assume that if the user has used a custom non-bonded force for the
    # vdW interactions then they will also have used a custom bond force to
    # define the 1-4 exclusion interactions so we need to track those also.
    custom_bond_forces = []

    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            normal_nonbonded_forces.append(force)
        elif isinstance(force, openmm.CustomNonbondedForce):
            custom_nonbonded_forces.append(force)
        elif isinstance(force, openmm.CustomBondForce):
            custom_bond_forces.append(force)

    if not (
        # vdW + electrostatics in a single built-in non-bonded force
        (len(normal_nonbonded_forces) == 1 and len(custom_nonbonded_forces) == 0)
        # OR electrostatics in a built-in non-bonded force, vdW in a custom
        # **non-bonded** force and vdW 1-4 interactions in a custom **bond** force
        or (
            len(normal_nonbonded_forces) == 1
            and len(custom_nonbonded_forces) == 1
            and len(custom_bond_forces) == 1
        )
        # OR all of the above sans any electrostatics.
        or (
            len(normal_nonbonded_forces) == 0
            and len(custom_nonbonded_forces) == 1
            and len(custom_bond_forces) == 1
        )
    ):
        raise NotImplementedError(
            "Currently only OpenMM systems that have:\n\n"
            "- vdW + electrostatics in a single built-in non-bonded force\n"
            "- electrostatics in a built-in non-bonded force, vdW in a custom "
            "**non-bonded** force and vdW 1-4 interactions in a custom **bond** "
            "force\n"
            "- all of the above sans any electrostatics\n\n"
            "are supported. Please raise an issue on the GitHub issue tracker if "
            "you'd like your use case supported."
        )

    nonbonded_force = (
        None if len(normal_nonbonded_forces) == 0 else normal_nonbonded_forces[0]
    )
    custom_nonbonded_force = (
        None if len(custom_nonbonded_forces) == 0 else custom_nonbonded_forces[0]
    )
    custom_bond_force = None if len(custom_bond_forces) == 0 else custom_bond_forces[0]

    if custom_nonbonded_force is not None and nonbonded_force is not None:
        for i in range(nonbonded_force.getNumParticles()):
            _, _, epsilon = nonbonded_force.getParticleParameters(i)

            if numpy.isclose(epsilon.value_in_unit(_KJ_MOL), 0.0):
                continue

            raise NotImplementedError(
                "The system contained both a `CustomNonbondedForce` and a "
                "`NonbondedForce` with non-zero epsilon parameters. Please raise "
                "an issue on the GitHub issue tracker if you'd like your use case "
                "supported."
            )

    return nonbonded_force, custom_nonbonded_force, custom_bond_force


def _add_electrostatics_lambda(
    nonbonded_force: openmm.NonbondedForce, alchemical_indices: list[set[int]]
):
    """Modifies a standard non-bonded force so that the charges are scaled
    by `lambda_electrostatics`. The alchemical-chemical interactions will be linearly
    scaled while the alchemical-alchemical interactions will be quadratically scaled.

    Args:
        nonbonded_force: The force to modify.
        alchemical_indices: The indices of the alchemical particles in the force.
    """

    assert (
        nonbonded_force.getNumGlobalParameters() == 0
    ), "the non-bonded force should not already contain global parameters"

    assert (
        nonbonded_force.getNumParticleParameterOffsets() == 0
        and nonbonded_force.getNumExceptionParameterOffsets() == 0
    ), "the non-bonded force should not already contain parameter offsets"

    nonbonded_force.addGlobalParameter(LAMBDA_ELECTROSTATICS, 1.0)

    alchemical_atom_indices = {i for values in alchemical_indices for i in values}

    for i in range(nonbonded_force.getNumParticles()):
        if i not in alchemical_atom_indices:
            continue

        charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
        nonbonded_force.setParticleParameters(i, charge * 0.0, sigma, epsilon)

        if numpy.isclose(charge.value_in_unit(_E), 0.0):
            # We don't need to scale already zero charges
            continue

        nonbonded_force.addParticleParameterOffset(
            LAMBDA_ELECTROSTATICS, i, charge, 0.0, 0.0
        )

    for i in range(nonbonded_force.getNumExceptions()):
        (
            idx_a,
            idx_b,
            charge_product,
            sigma,
            epsilon,
        ) = nonbonded_force.getExceptionParameters(i)

        if (
            idx_a not in alchemical_atom_indices
            and idx_b not in alchemical_atom_indices
        ):
            continue

        if numpy.isclose(charge_product.value_in_unit(_E * _E), 0.0):
            # As above, we don't need to scale already zero charge exceptions
            continue

        assert (
            idx_a in alchemical_atom_indices and idx_b in alchemical_atom_indices
        ), "both atoms in an exception should be alchemically transformed"

        nonbonded_force.setExceptionParameters(
            i, idx_a, idx_b, charge_product * 0.0, sigma, epsilon
        )
        nonbonded_force.addExceptionParameterOffset(
            LAMBDA_ELECTROSTATICS, i, charge_product, 0.0, 0.0
        )


def _add_lj_vdw_lambda(
    original_force: openmm.NonbondedForce,
    alchemical_indices: list[set[int]],
    persistent_indices: list[set[int]],
    custom_alchemical_potential: str | None,
) -> tuple[openmm.CustomNonbondedForce, openmm.CustomNonbondedForce]:
    """Modifies a standard non-bonded force so that only the interactions between
    persistent (chemical) particles are retained, and splits out all intermolecular
    alchemical (both chemical-alchemical and alchemical-alchemical) interactions
    and all intramolecular alchemical-alchemical interactions into two new
    custom non-bonded forces.

    Notes:
        * By default the custom non-bonded force will use a soft-core version of the
          LJ potential with a-b-c of 1-1-6 and alpha=0.5 that can be scaled by a
          global `lambda_sterics` parameter.
        * The alchemical-alchemical intramolecular custom non-bonded force will use
          ``NoCutoff` as the interaction if ``original_force`` also uses ``NoCutoff``
          or ``CutoffNonPeriodic`` otherwise even if ``original_force`` uses
          ``CutoffPeriodic``. This ensures that the decoupled state truly
          corresponds to a set of alchemical molecules in vacuum with no periodic
          effects.

    Args:
        original_force: The force to modify.
        alchemical_indices: The indices of the alchemical particles in the force.
        persistent_indices: The indices of the chemical particles in the force.
        custom_alchemical_potential: A custom expression to use for the potential
            energy function that describes the chemical-alchemical intermolecular
            interactions. See the Notes for information about the default value.

            The expression **must** include ``"lambda_sterics"``.

    Returns:
        A custom non-bonded force that contains all chemical-alchemical and
        **intermolecular** alchemical-alchemical interactions, and a custom
        non-bonded force that contains the intermolecular interactions of the
        alchemical molecules excluding any 1-2, 1-3, and 1-4 interactions which are
        still handled by the original force.
    """

    custom_nonbonded_template = openmm.CustomNonbondedForce("")
    custom_nonbonded_template.setNonbondedMethod(
        original_force.getNonbondedMethod()
        if int(original_force.getNonbondedMethod()) not in {3, 4, 5}
        else openmm.CustomNonbondedForce.CutoffPeriodic
    )
    custom_nonbonded_template.setCutoffDistance(original_force.getCutoffDistance())
    custom_nonbonded_template.setSwitchingDistance(
        original_force.getSwitchingDistance()
    )
    custom_nonbonded_template.setUseSwitchingFunction(
        original_force.getUseSwitchingFunction()
    )
    custom_nonbonded_template.setUseLongRangeCorrection(
        original_force.getUseDispersionCorrection()
    )

    custom_nonbonded_template.addPerParticleParameter("sigma")
    custom_nonbonded_template.addPerParticleParameter("epsilon")

    alchemical_atom_indices = {i for values in alchemical_indices for i in values}
    persistent_atom_indices = {i for values in persistent_indices for i in values}

    original_parameters = {}

    for idx in range(original_force.getNumParticles()):
        charge, sigma, epsilon = original_force.getParticleParameters(idx)

        custom_nonbonded_template.addParticle([sigma, epsilon])

        if idx not in alchemical_atom_indices:
            continue

        # The vdW intermolecular alchemical interactions will be handled in a custom
        # nonbonded force below, while the intramolecular interactions will be
        # converted to exceptions so we zero them out here.
        original_parameters[idx] = (sigma, epsilon)
        original_force.setParticleParameters(idx, charge, sigma, epsilon * 0)

    for idx in range(original_force.getNumExceptions()):
        idx_a, idx_b, *_ = original_force.getExceptionParameters(idx)
        # Let the exceptions be handled by the original force as we don't intend to
        # annihilate those while switching off the intermolecular vdW interactions
        custom_nonbonded_template.addExclusion(idx_a, idx_b)

    # Make sure that each alchemical molecule interacts with each chemical molecule
    if custom_alchemical_potential is None:
        custom_alchemical_potential = LJ_POTENTIAL_BEUTLER + LORENTZ_BERTHELOT

    aa_na_custom_nonbonded_force = copy.deepcopy(custom_nonbonded_template)
    aa_na_custom_nonbonded_force.addGlobalParameter(LAMBDA_STERICS, 1.0)
    aa_na_custom_nonbonded_force.setEnergyFunction(custom_alchemical_potential)

    aa_na_custom_nonbonded_force.addInteractionGroup(
        alchemical_atom_indices, persistent_atom_indices
    )
    # and each alchemical molecule so that things like ion pair interactions are
    # disabled
    for pair in itertools.combinations(alchemical_indices, r=2):
        aa_na_custom_nonbonded_force.addInteractionGroup({*pair[0]}, {*pair[1]})

    # Make sure that each alchemical molecule can also interact with themselves
    aa_aa_custom_nonbonded_force = copy.deepcopy(custom_nonbonded_template)
    aa_aa_custom_nonbonded_force.setEnergyFunction(LJ_POTENTIAL + LORENTZ_BERTHELOT)
    aa_aa_custom_nonbonded_force.setNonbondedMethod(
        # Use a ``CutoffNonPeriodic`` to ensure solutes don't interact with
        # periodic copies of themselves
        openmm.NonbondedForce.NoCutoff
        if custom_nonbonded_template.getNonbondedMethod()
        == openmm.NonbondedForce.NoCutoff
        else openmm.NonbondedForce.CutoffNonPeriodic
    )

    for atom_indices in alchemical_indices:
        aa_aa_custom_nonbonded_force.addInteractionGroup(atom_indices, atom_indices)

    return aa_na_custom_nonbonded_force, aa_aa_custom_nonbonded_force


def _add_custom_vdw_lambda(
    original_force: openmm.CustomNonbondedForce,
    alchemical_indices: list[set[int]],
    persistent_indices: list[set[int]],
    custom_alchemical_potential: str | None,
) -> tuple[openmm.CustomNonbondedForce, openmm.CustomNonbondedForce]:
    """Modifies a custom non-bonded force so that only the interactions between
    persistent (chemical) particles and all intramolecular interactions (including
    those in alchemical molecules) are retained, and splits out all intermolecular
    alchemical (both chemical-alchemical and alchemical-alchemical) interactions
    and all intramolecular alchemical-alchemical interactions into two new
    custom non-bonded forces.

    Notes:
        * By default the chemical-alchemical custom non-bonded force will use a
          modified energy expression so that it can be linearly scaled by a global
          `lambda_sterics` parameter.
        * The alchemical-alchemical intramolecular custom non-bonded force will use
          ``NoCutoff` as the interaction if ``original_force`` also uses ``NoCutoff``
          or ``CutoffNonPeriodic`` otherwise even if ``original_force`` uses
          ``CutoffPeriodic``. This ensures that the decoupled state truly
          corresponds to a set of alchemical molecules in vacuum with no periodic
          effects.

    Args:
        original_force: The force to modify.
        alchemical_indices: The indices of the alchemical particles in the force.
        persistent_indices: The indices of the chemical particles in the force.
        custom_alchemical_potential: A custom expression to use for the potential
            energy function that describes the chemical-alchemical intermolecular
            interactions. See the Notes for information about the default value.

            The expression **must** include ``"lambda_sterics"``.

    Returns:
        A custom non-bonded forces that contains all chemical-alchemical and
        intermolecular alchemical-alchemical interactions, and one containing all of
        the intramolecular alchemical-alchemical interactions excluding any 1-2, 1-3
        and 1-4 interactions which should be instead handled by an un-scaled custom
        bond force.
    """

    assert (
        original_force.getNumInteractionGroups() == 0
    ), "the custom force should not contain any interaction groups"

    custom_nonbonded_template = copy.deepcopy(original_force)

    alchemical_atom_indices = {i for values in alchemical_indices for i in values}
    persistent_atom_indices = {i for values in persistent_indices for i in values}

    assert alchemical_atom_indices.isdisjoint(persistent_atom_indices)

    # Modify the original force so it only targets the chemical-chemical interactions
    original_force.addInteractionGroup(persistent_atom_indices, persistent_atom_indices)

    # Make sure that each alchemical molecule interacts with each chemical molecule,
    aa_na_custom_nonbonded_force = copy.deepcopy(custom_nonbonded_template)
    aa_na_custom_nonbonded_force.addGlobalParameter(LAMBDA_STERICS, 1.0)

    if custom_alchemical_potential is None:
        energy_expression = aa_na_custom_nonbonded_force.getEnergyFunction().split(";")

        for i, expression in enumerate(energy_expression):
            if "=" not in expression:
                energy_expression[i] = f"{LAMBDA_STERICS}*({expression})"
                break

        custom_alchemical_potential = ";".join(energy_expression)

    aa_na_custom_nonbonded_force.setEnergyFunction(custom_alchemical_potential)

    aa_na_custom_nonbonded_force.addInteractionGroup(
        alchemical_atom_indices, persistent_atom_indices
    )
    # and all other alchemical molecules so that things like ion pair interactions
    # are included
    for pair in itertools.combinations(alchemical_indices, r=2):
        aa_na_custom_nonbonded_force.addInteractionGroup({*pair[0]}, {*pair[1]})

    # Make sure that each alchemical molecule can also interact with themselves
    # excluding any 1-2, 1-3, and 1-4 interactions
    aa_aa_custom_nonbonded_force = copy.deepcopy(custom_nonbonded_template)
    aa_aa_custom_nonbonded_force.setNonbondedMethod(
        openmm.CustomNonbondedForce.NoCutoff
        if custom_nonbonded_template.getNonbondedMethod()
        == openmm.CustomNonbondedForce.NoCutoff
        else openmm.CustomNonbondedForce.CutoffNonPeriodic
    )

    for atom_indices in alchemical_indices:
        aa_aa_custom_nonbonded_force.addInteractionGroup(atom_indices, atom_indices)

    return aa_na_custom_nonbonded_force, aa_aa_custom_nonbonded_force


def apply_fep(
    system: openmm.System,
    alchemical_indices: list[set[int]],
    persistent_indices: list[set[int]],
    custom_alchemical_potential: str | None = None,
) -> openmm.System:
    """Generate a system whereby a number of the molecules can be alchemically
    transformed from a base chemical system.

    Notes:
        * Currently only OpenMM systems that have:

          - vdW + electrostatics in a single built-in non-bonded force
          - electrostatics in a built-in non-bonded force, vdW in a custom
            **non-bonded** force and vdW 1-4 interactions in a custom **bond** force
          - all of the above sans any electrostatics

          are supported.
        * By default a soft-core version of the LJ potential with a-b-c of 1-1-6
          and alpha=0.5 that can be scaled by a global `lambda_sterics` parameter
          will be used for alchemical-chemical vdW interactions embedded in an
          OpenMM ``NonbondedForce`` while the energy expression set on a
          ``CustomNonbondedForce`` will be be modified to have the form
          ``"lambda_sterics*({original_expression})"``.

    Args:
        system: The chemical system to generate the alchemical system from
        alchemical_indices: The atom indices corresponding to each molecule that
            should be alchemically transformable. The atom indices **must**
            correspond to  **all** atoms / v-sites in each molecule as alchemically
            transforming part of a molecule is not supported.
        persistent_indices: The atom indices corresponding to each molecule that
            should **not** be alchemically transformable.
        custom_alchemical_potential: A custom expression to use for the potential
            energy function that describes the chemical-alchemical intermolecular
            interactions. See the Notes for information about the default value.

            The expression **must** include ``"lambda_sterics"``.
    """

    system = copy.deepcopy(system)

    (
        nonbonded_force,
        custom_nonbonded_force,
        custom_bond_force,
    ) = _find_nonbonded_forces(system)

    if nonbonded_force is not None:
        _add_electrostatics_lambda(nonbonded_force, alchemical_indices)

    if custom_nonbonded_force is not None:
        for alchemical_force in _add_custom_vdw_lambda(
            custom_nonbonded_force,
            alchemical_indices,
            persistent_indices,
            custom_alchemical_potential,
        ):
            system.addForce(alchemical_force)

    elif nonbonded_force is not None:
        for alchemical_force in _add_lj_vdw_lambda(
            nonbonded_force,
            alchemical_indices,
            persistent_indices,
            custom_alchemical_potential,
        ):
            system.addForce(alchemical_force)

    else:
        raise NotImplementedError

    return system


def set_fep_lambdas(
    context: openmm.Context,
    lambda_sterics: float | None = None,
    lambda_electrostatics: float | None = None,
):
    """Set the values of the alchemical lambdas on an OpenMM context.

    Args:
        context: The context to update.
        lambda_sterics: The (optional) value of the steric lambda.
        lambda_electrostatics: The (optional) value of the electrostatics lambda.
    """

    if lambda_sterics is not None:
        assert (
            0.0 <= lambda_sterics <= 1.0
        ), f"`{LAMBDA_STERICS}` must be between 0 and 1"
        context.setParameter(LAMBDA_STERICS, lambda_sterics)

    if lambda_electrostatics is not None:
        assert (
            0.0 <= lambda_electrostatics <= 1.0
        ), f"`{LAMBDA_ELECTROSTATICS}` must be between 0 and 1"

        context.setParameter(LAMBDA_ELECTROSTATICS, lambda_electrostatics)
