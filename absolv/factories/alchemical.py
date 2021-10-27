import copy
import itertools
from typing import List, Optional, Set, Tuple

import numpy
import openmm
from openmm import unit


def _lorentz_berthelot() -> str:
    return "sigma = 0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2);"


def _lj_potential() -> str:
    return "4.0*epsilon*x*(x-1.0); x = (sigma/r)^6;" + _lorentz_berthelot()


def _soft_core_lj_potential() -> str:
    return (
        "lambda_sterics*4.0*epsilon*x*(x-1.0);"
        "x = (sigma/reff_sterics)^6;"
        "reff_sterics = sigma*(0.5*(1.0-lambda_sterics) + (r/sigma)^6)^(1/6);"
    ) + _lorentz_berthelot()


class OpenMMAlchemicalFactory:
    """A factory that aids in transforming existing OpenMM systems into ones whose
    non-bonded interactions can be alchemically disabled using global context parameters.
    """

    @classmethod
    def _find_v_sites(cls, system: openmm.System, atom_indices: List[Set[int]]):
        """Finds any virtual sites in the system and ensures their indices get appended
        to the atom index list.

        Args:
            system: The system that may contain v-sites.
            atom_indices: A list of per-molecule atom indices to add the virtual site
                indices to.
        """

        atom_to_molecule_index = {
            atom_index: i
            for i, indices in enumerate(atom_indices)
            for atom_index in indices
        }

        for i in range(system.getNumParticles()):

            if not system.isVirtualSite(i):

                assert i in atom_to_molecule_index, "the atom indices are invalid"
                continue

            v_site = system.getVirtualSite(i)
            parent_atom_index = v_site.getParticle(0)

            molecule_index = atom_to_molecule_index[parent_atom_index]
            atom_indices[molecule_index].add(i)

    @classmethod
    def _find_nonbonded_forces(
        cls, system: openmm.System
    ) -> Tuple[
        Optional[openmm.NonbondedForce],
        Optional[openmm.CustomNonbondedForce],
        Optional[openmm.CustomBondForce],
    ]:
        """Attempts to find the forces that describe the non-bonded (both vdW and
        electrostatic) interactions in the system.

        Args:
            system: The OpenMM system containing the forces.

        Notes:
            * Currently only OpenMM systems that have:

              - vdW + electrostatics in a single built-in LJ force
              - vdW + electrostatics in a single built-in LJ force and vdW 1-4
                interactions in a custom **bond** force
              - electrostatics in a built-in LJ force, vdW in a custom **non-bonded**
                force and vdW 1-4 interactions in a custom **bond** force
              - all of the above sans any electrostatics.
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
            # vdW + electrostatics in a single built-in LJ force
            (len(normal_nonbonded_forces) == 1 and len(custom_nonbonded_forces) == 0)
            # OR vdW + electrostatics in a single built-in LJ force and vdW 1-4
            #    interactions in a custom **bond** force
            or (len(normal_nonbonded_forces) == 1 and len(custom_bond_forces) == 1)
            # OR electrostatics in a built-in LJ force, vdW in a custom **non-bonded**
            #    force and vdW 1-4 interactions in a custom **bond** force
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
                "- vdW + electrostatics in a single built-in LJ force\n"
                "- vdW + electrostatics in a single built-in LJ force and vdW 1-4 "
                "interactions in a custom **bond** force\n"
                "- electrostatics in a built-in LJ force, vdW in a custom **non-"
                "bonded** force and vdW 1-4 interactions in a custom **bond** force\n"
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
        custom_bond_force = (
            None if len(custom_bond_forces) == 0 else custom_bond_forces[0]
        )

        if custom_nonbonded_force is not None and nonbonded_force is not None:

            for i in range(nonbonded_force.getNumParticles()):

                _, _, epsilon = nonbonded_force.getParticleParameters(i)

                if numpy.isclose(epsilon.value_in_unit(unit.kilojoule_per_mole), 0.0):
                    continue

                raise NotImplementedError(
                    "The system contained both a `CustomNonbondedForce` and a "
                    "`NonbondedForce` with non-zero epsilon parameters. Please raise "
                    "an issue on the GitHub issue tracker if you'd like your use case "
                    "supported."
                )

        return nonbonded_force, custom_nonbonded_force, custom_bond_force

    @classmethod
    def _add_electrostatics_lambda(
        cls,
        nonbonded_force: openmm.NonbondedForce,
        alchemical_indices: List[Set[int]],
    ):
        """Modifies a standard non-bonded force so that the charges are linearly scaled
        by `lambda_electrostatics`.

        Args:
            nonbonded_force: The force to modifiy.
            alchemical_indices: The indices of the alchemical particles in the force.
        """

        global_parameter_names = {
            nonbonded_force.getGlobalParameterName(i)
            for i in range(nonbonded_force.getNumGlobalParameters())
        }
        assert (
            "lambda_electrostatics" not in global_parameter_names
        ), "electrostatics lambda already present"
        assert (
            nonbonded_force.getNumParticleParameterOffsets() == 0
            and nonbonded_force.getNumExceptionParameterOffsets() == 0
        ), "the non-bonded force should not already contain parameter offsets"

        nonbonded_force.addGlobalParameter("lambda_electrostatics", 1.0)

        alchemical_atom_indices = {i for values in alchemical_indices for i in values}

        for i in range(nonbonded_force.getNumParticles()):

            if i not in alchemical_atom_indices:
                continue

            charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
            nonbonded_force.setParticleParameters(i, charge * 0.0, sigma, epsilon)

            if numpy.isclose(charge.value_in_unit(unit.elementary_charge), 0.0):
                # We don't need to scale already zero charges
                continue

            # TODO: Does it matter if we don't scale by sqrt lambda here?
            nonbonded_force.addParticleParameterOffset(
                "lambda_electrostatics", i, charge, 0.0, 0.0
            )

        for i in range(nonbonded_force.getNumExceptions()):

            (
                index_a,
                index_b,
                charge_product,
                sigma,
                epsilon,
            ) = nonbonded_force.getExceptionParameters(i)

            if (
                index_a not in alchemical_atom_indices
                and index_b not in alchemical_atom_indices
            ):
                continue

            if numpy.isclose(
                charge_product.value_in_unit(
                    unit.elementary_charge * unit.elementary_charge
                ),
                0.0,
            ):
                # As above we don't need to scale already zero charge exceptions
                continue

            assert (
                index_a in alchemical_atom_indices
                and index_b in alchemical_atom_indices
            ), "both atoms in an exception should be alchemically transformed"

            nonbonded_force.setExceptionParameters(
                i, index_a, index_b, charge_product * 0.0, sigma, epsilon
            )
            nonbonded_force.addExceptionParameterOffset(
                "lambda_electrostatics", i, charge_product, 0.0, 0.0
            )

    @classmethod
    def _add_lj_vdw_lambda(
        cls,
        original_force: openmm.NonbondedForce,
        alchemical_indices: List[Set[int]],
        persistent_indices: List[Set[int]],
    ) -> Tuple[openmm.CustomNonbondedForce, openmm.CustomNonbondedForce]:
        """Modifies a standard non-bonded force so that only the interactions between
        persistent (chemical) particles and all 1-4 interactions are retained, and splits
        out all alchemical (both chemical-alchemical and alchemical-alchemical)
        interactions into two custom non-bonded forces.

        Notes:
            The chemical-alchemical custom non-bonded force will use a soft-core
            version of the LJ potential with a-b-c of 1-1-6 and alpha=0.5 that can be
            scaled by a global `lambda_sterics` parameter.

        Args:
            original_force: The force to modify.
            alchemical_indices: The indices of the alchemical particles in the force.
            persistent_indices: The indices of the chemical particles in the force.

        Returns:
            Custom non-bonded forces that contain all of the chemical-alchemical and
            alchemical-alchemical interactions, not including any 1-2, 1-3 and 1-4
            interactions which should be handled by either the built-in non-bonded force
            or an unscaled custom bond force.
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
            False  # original_force.getUseDispersionCorrection()
        )

        custom_nonbonded_template.addPerParticleParameter("sigma")
        custom_nonbonded_template.addPerParticleParameter("epsilon")

        alchemical_atom_indices = {i for values in alchemical_indices for i in values}
        persistent_atom_indices = {i for values in persistent_indices for i in values}

        for index in range(original_force.getNumParticles()):

            charge, sigma, epsilon = original_force.getParticleParameters(index)

            custom_nonbonded_template.addParticle([sigma, epsilon])

            if index not in alchemical_atom_indices:
                continue

            original_force.setParticleParameters(index, charge, sigma, epsilon * 0)

        for index in range(original_force.getNumExceptions()):

            index_a, index_b, *_ = original_force.getExceptionParameters(index)
            # Let the exceptions be handled by the original force as we don't intent to
            # annihilate those while switching off the intermolecular vdW interactions
            custom_nonbonded_template.addExclusion(index_a, index_b)

        # Make sure that each alchemical molecule interacts with each chemical molecule
        aa_na_custom_nonbonded_force = copy.deepcopy(custom_nonbonded_template)
        aa_na_custom_nonbonded_force.addGlobalParameter("lambda_sterics", 1.0)
        aa_na_custom_nonbonded_force.setEnergyFunction(_soft_core_lj_potential())

        aa_na_custom_nonbonded_force.addInteractionGroup(
            alchemical_atom_indices, persistent_atom_indices
        )

        # Make sure that each alchemical molecule can also interact with themselves
        # excluding any 1-2, 1-3, and 1-4 interactions
        aa_aa_custom_nonbonded_force = copy.deepcopy(custom_nonbonded_template)
        aa_aa_custom_nonbonded_force.setEnergyFunction(_lj_potential())

        for atom_indices in alchemical_indices:
            aa_aa_custom_nonbonded_force.addInteractionGroup(atom_indices, atom_indices)
        # and that each alchemical molecule interacts with each other alchemical molecule
        for pair in itertools.combinations(alchemical_indices, r=2):
            aa_aa_custom_nonbonded_force.addInteractionGroup({*pair[0]}, {*pair[1]})

        return aa_na_custom_nonbonded_force, aa_aa_custom_nonbonded_force

    @classmethod
    def _add_custom_vdw_lambda(
        cls,
        original_force: openmm.CustomNonbondedForce,
        alchemical_indices: List[Set[int]],
        persistent_indices: List[Set[int]],
    ) -> Tuple[openmm.CustomNonbondedForce, openmm.CustomNonbondedForce]:
        """Modifies a custom non-bonded force so that only the interactions between
        persistent (chemical) particles are retained, and splits out all alchemical
        (both chemical-alchemical and alchemical-alchemical) interactions into two
        custom non-bonded forces.

        Notes:
            The alchemical-alchemical custom non-bonded force will use a modified energy
            expression so that it can be linearly scaled by a global `lambda_sterics`
            parameter.

        Args:
            original_force: The force to modify.
            alchemical_indices: The indices of the alchemical particles in the force.
            persistent_indices: The indices of the chemical particles in the force.

        Returns:
            Custom non-bonded forces that contain all of the chemical-alchemical and
            alchemical-alchemical interactions, not including any 1-2, 1-3 and 1-4
            interactions which should be instead handled by an un-modified custom bond
            force.
        """

        assert original_force.getNumInteractionGroups() == 0, (
            "the custom force should not contain " "any interaction groups"
        )

        custom_nonbonded_template = copy.deepcopy(original_force)

        alchemical_atom_indices = {i for values in alchemical_indices for i in values}
        persistent_atom_indices = {i for values in persistent_indices for i in values}

        # Modify the original force so it only targets the chemical-chemical interactions
        original_force.addInteractionGroup(
            persistent_atom_indices, persistent_atom_indices
        )

        # Make sure that each alchemical molecule interacts with each chemical molecule,
        aa_na_custom_nonbonded_force = copy.deepcopy(custom_nonbonded_template)
        aa_na_custom_nonbonded_force.addGlobalParameter("lambda_sterics", 1.0)

        energy_expression = aa_na_custom_nonbonded_force.getEnergyFunction().split(";")
        for i, expression in enumerate(energy_expression):
            if "=" not in expression:
                energy_expression[i] = f"lambda_sterics*({expression})"
                break
        aa_na_custom_nonbonded_force.setEnergyFunction(";".join(energy_expression))

        aa_na_custom_nonbonded_force.addInteractionGroup(
            alchemical_atom_indices, persistent_atom_indices
        )

        # Make sure that each alchemical molecule can also interact with themselves
        # excluding any 1-2, 1-3, and 1-4 interactions
        aa_aa_custom_nonbonded_force = copy.deepcopy(custom_nonbonded_template)

        for atom_indices in alchemical_indices:
            aa_aa_custom_nonbonded_force.addInteractionGroup(atom_indices, atom_indices)
        for pair in itertools.combinations(alchemical_indices, r=2):
            aa_aa_custom_nonbonded_force.addInteractionGroup({*pair[0]}, {*pair[1]})

        return aa_na_custom_nonbonded_force, aa_aa_custom_nonbonded_force

    @classmethod
    def generate(
        cls,
        system: openmm.System,
        alchemical_indices: List[Set[int]],
        persistent_indices: List[Set[int]],
    ) -> openmm.System:
        """Generate a system whereby a number of the molecules can be alchemically
         transformed from a base chemical system.

        Args:
            system: The chemical system to generate the alchemical system from
            alchemical_indices: The atom indices corresponding to each molecule that
                should be alchemically transformable. The atom indices **must**
                correspond to  **all** atoms in each molecule as alchemically
                transforming part of a molecule is not supported.
            persistent_indices: The atom indices corresponding to each molecule that
                should **not** be alchemically transformable.
        """

        system = copy.deepcopy(system)

        # Make sure we track v-sites attached to any solutes that may be alchemically
        # turned off. We do this as a post-process step as the OpenFF toolkit does not
        # currently expose a clean way to access this information.
        atom_indices = alchemical_indices + persistent_indices
        cls._find_v_sites(system, atom_indices)

        alchemical_indices = atom_indices[: len(alchemical_indices)]
        persistent_indices = atom_indices[len(alchemical_indices) :]

        (
            nonbonded_force,
            custom_nonbonded_force,
            custom_bond_force,
        ) = cls._find_nonbonded_forces(system)

        if nonbonded_force is not None:
            cls._add_electrostatics_lambda(nonbonded_force, alchemical_indices)

        if custom_nonbonded_force is not None:

            for alchemical_force in cls._add_custom_vdw_lambda(
                custom_nonbonded_force, alchemical_indices, persistent_indices
            ):
                system.addForce(alchemical_force)

        elif nonbonded_force is not None:

            for alchemical_force in cls._add_lj_vdw_lambda(
                nonbonded_force, alchemical_indices, persistent_indices
            ):
                system.addForce(alchemical_force)

        else:
            raise NotImplementedError

        return system
