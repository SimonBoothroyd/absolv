import copy

import numpy
import openmm
import pytest
from openmm import unit

from absolv.factories.alchemical import (
    OpenMMAlchemicalFactory,
    lj_potential,
    lorentz_berthelot,
    soft_core_lj_potential,
)
from absolv.tests import is_close


class TestOpenMMAlchemicalFactory:
    def test_find_v_sites(self):
        """Ensure that v-sites are correctly detected from an OMM system and assigned
        to the right parent molecule."""

        # Construct a mock system of V A A A V A A where (0, 5, 6), (3,), (4, 1, 2)
        # are the core molecules.
        system = openmm.System()

        for i in range(7):
            system.addParticle(1.0)

        system.setVirtualSite(0, openmm.TwoParticleAverageSite(5, 6, 0.5, 0.5))
        system.setVirtualSite(4, openmm.TwoParticleAverageSite(1, 2, 0.5, 0.5))

        atom_indices = [{0, 1}, {2}, {3, 4}]

        particle_indices = OpenMMAlchemicalFactory._find_v_sites(system, atom_indices)

        assert particle_indices == [{1, 2, 4}, {3}, {0, 5, 6}]

    def test_find_nonbonded_forces_lj_only(self, aq_nacl_lj_system):

        (
            nonbonded_force,
            custom_nonbonded_force,
            custom_bond_force,
        ) = OpenMMAlchemicalFactory._find_nonbonded_forces(aq_nacl_lj_system)

        assert nonbonded_force is not None
        assert isinstance(nonbonded_force, openmm.NonbondedForce)

        assert nonbonded_force.getNumParticles() == 10  # Na + Cl + 2 x O with 2 v-site

        assert custom_nonbonded_force is None
        assert custom_bond_force is None

    def test_find_nonbonded_forces_custom(self, aq_meoh_de_system):

        expected_types = (
            openmm.NonbondedForce,
            openmm.CustomNonbondedForce,
            openmm.CustomBondForce,
        )
        found_forces = OpenMMAlchemicalFactory._find_nonbonded_forces(aq_meoh_de_system)

        assert all(
            force is not None and isinstance(force, expected_type)
            for force, expected_type in zip(found_forces, expected_types)
        )

        nonbonded_force, custom_nonbonded_force, custom_bond_force = found_forces

        assert (
            nonbonded_force.getNumParticles() == 14
        )  # C1H4O1 + 2 x O1H2 with 2 v-site
        assert (
            custom_nonbonded_force.getNumParticles() == 14
        )  # C1H4O1 + 2 x O1H2 with 2 v-site

        assert custom_bond_force.getNumBonds() == 3  # 4 x H - - H 1-4 interactions

    def test_add_electrostatics_lambda(self, aq_nacl_lj_nonbonded):

        nonbonded_force = copy.deepcopy(aq_nacl_lj_nonbonded)

        OpenMMAlchemicalFactory._add_electrostatics_lambda(nonbonded_force, [{0}, {1}])

        assert nonbonded_force.getNumGlobalParameters() == 1
        assert nonbonded_force.getGlobalParameterName(0) == "lambda_electrostatics"
        numpy.isclose(nonbonded_force.getGlobalParameterDefaultValue(0), 1.0)

        assert nonbonded_force.getNumParticleParameterOffsets() == 2

        for i in range(2):

            (
                parameter,
                index,
                charge_scale,
                sigma_scale,
                epsilon_scale,
            ) = nonbonded_force.getParticleParameterOffset(i)

            assert parameter == "lambda_electrostatics"
            assert index == i

            assert numpy.isclose(charge_scale, [1, -1][i])
            assert numpy.isclose(epsilon_scale, 0.0)
            assert numpy.isclose(sigma_scale, 0.0)

            charge, *_ = nonbonded_force.getParticleParameters(i)

            assert numpy.isclose(charge.value_in_unit(unit.elementary_charge), 0.0)

        for i in [3, 4, 6, 7, 8, 9]:  # H + v-sites

            charge, *_ = nonbonded_force.getParticleParameters(i)
            assert not numpy.isclose(charge.value_in_unit(unit.elementary_charge), 0.0)

    @pytest.mark.parametrize("custom_potential", [None, "lambda_sterics*2.0"])
    def test_add_lj_vdw_lambda(self, aq_nacl_lj_nonbonded, custom_potential):

        nonbonded_force = copy.deepcopy(aq_nacl_lj_nonbonded)

        custom_forces = OpenMMAlchemicalFactory._add_lj_vdw_lambda(
            nonbonded_force, [{0}, {1}], [{2, 3, 4}, {5, 6, 7}], custom_potential
        )

        assert nonbonded_force.getNonbondedMethod() == openmm.NonbondedForce.PME
        assert nonbonded_force.getNumGlobalParameters() == 0

        # Make sure the solute interactions have been disabled while the solvent ones.
        # are still present
        assert all(
            is_close(
                nonbonded_force.getParticleParameters(i)[2],
                0.0 * unit.kilojoule_per_mole,
            )
            for i in [0, 1]
        )
        assert all(
            is_close(
                nonbonded_force.getParticleParameters(i)[2],
                aq_nacl_lj_nonbonded.getParticleParameters(i)[2],
            )
            for i in range(2, nonbonded_force.getNumParticles())
        )

        # Make sure the built-in exceptions remain unchanged.
        assert all(
            aq_nacl_lj_nonbonded.getExceptionParameters(i)
            == nonbonded_force.getExceptionParameters(i)
            for i in range(nonbonded_force.getNumExceptions())
        )

        assert isinstance(custom_forces[0], openmm.CustomNonbondedForce)
        assert isinstance(custom_forces[1], openmm.CustomNonbondedForce)

        aa_na_custom_force, aa_aa_custom_force = custom_forces

        # Make sure only the alchemical-chemical interactions are transformed
        if custom_potential is None:
            assert (
                aa_na_custom_force.getEnergyFunction()
                == soft_core_lj_potential() + lorentz_berthelot()
            )
        else:
            assert aa_na_custom_force.getEnergyFunction() == custom_potential

        assert aa_na_custom_force.getNumGlobalParameters() == 1
        assert aa_na_custom_force.getGlobalParameterName(0) == "lambda_sterics"
        assert (
            aa_na_custom_force.getNonbondedMethod()
            == openmm.CustomNonbondedForce.CutoffPeriodic
        )

        assert (
            aa_aa_custom_force.getEnergyFunction()
            == lj_potential() + lorentz_berthelot()
        )
        assert aa_aa_custom_force.getNumGlobalParameters() == 0
        assert (
            aa_aa_custom_force.getNonbondedMethod()
            == openmm.CustomNonbondedForce.CutoffNonPeriodic
        )

        # Make sure the alchemical-chemical interaction groups are correctly set-up
        assert aa_na_custom_force.getNumInteractionGroups() == 2
        assert aa_na_custom_force.getInteractionGroupParameters(0) == [
            (0, 1),
            (2, 3, 4, 5, 6, 7),
        ]
        assert aa_na_custom_force.getInteractionGroupParameters(1) == [(0,), (1,)]

        assert aa_aa_custom_force.getNumInteractionGroups() == 2
        assert aa_aa_custom_force.getInteractionGroupParameters(0) == [(0,), (0,)]
        assert aa_aa_custom_force.getInteractionGroupParameters(1) == [(1,), (1,)]

    @pytest.mark.parametrize("custom_potential", [None, "lambda_sterics*2.0"])
    def test_add_custom_vdw_lambda(self, aq_meoh_de_nonbonded, custom_potential):

        original_force = copy.deepcopy(aq_meoh_de_nonbonded)

        custom_forces = OpenMMAlchemicalFactory._add_custom_vdw_lambda(
            original_force,
            [{0, 1, 2, 3, 4, 5}],
            [{6, 7, 8}, {9, 10, 11}],
            custom_potential,
        )

        assert (
            original_force.getNumGlobalParameters()
            == aq_meoh_de_nonbonded.getNumGlobalParameters()
        )

        # Make sure the solute interactions have been disabled while the solvent ones.
        # are still present
        assert original_force.getNumInteractionGroups() == 1
        assert original_force.getInteractionGroupParameters(0) == [
            (6, 7, 8, 9, 10, 11),
            (6, 7, 8, 9, 10, 11),
        ]

        assert all(
            isinstance(force, openmm.CustomNonbondedForce) for force in custom_forces
        )

        aa_na_custom_force, aa_aa_custom_force = custom_forces

        # Make sure only the alchemical-chemical interactions are transformed
        if custom_potential is None:
            expected_expression = (
                f"lambda_sterics*({aq_meoh_de_nonbonded.getEnergyFunction()}"
            ).replace("AttractionExp;", "AttractionExp);")

            assert aa_na_custom_force.getEnergyFunction() == expected_expression
        else:
            assert aa_na_custom_force.getEnergyFunction() == custom_potential

        assert (
            aa_na_custom_force.getNumGlobalParameters()
            == aq_meoh_de_nonbonded.getNumGlobalParameters() + 1
        )
        assert (
            aa_na_custom_force.getGlobalParameterName(
                aq_meoh_de_nonbonded.getNumGlobalParameters()
            )
            == "lambda_sterics"
        )

        assert (
            aa_aa_custom_force.getEnergyFunction()
            == aq_meoh_de_nonbonded.getEnergyFunction()
        )
        assert (
            aa_aa_custom_force.getNumGlobalParameters()
            == aq_meoh_de_nonbonded.getNumGlobalParameters()
        )

        # Make sure the alchemical-chemical interaction groups are correctly set-up
        assert aa_na_custom_force.getNumInteractionGroups() == 1
        assert aa_na_custom_force.getInteractionGroupParameters(0) == [
            (0, 1, 2, 3, 4, 5),
            (6, 7, 8, 9, 10, 11),
        ]

        assert aa_aa_custom_force.getNumInteractionGroups() == 1
        assert aa_aa_custom_force.getInteractionGroupParameters(0) == [
            (0, 1, 2, 3, 4, 5),
            (0, 1, 2, 3, 4, 5),
        ]

        assert (
            aa_aa_custom_force.getNonbondedMethod()
            == openmm.CustomNonbondedForce.CutoffNonPeriodic
        )

    def test_generate_electrostatic_and_lj(self, aq_nacl_lj_system, monkeypatch):

        original_system = copy.deepcopy(aq_nacl_lj_system)

        alchemical_system = OpenMMAlchemicalFactory.generate(
            original_system, [{0}, {1}], [{2, 3, 4}, {5, 6, 7}]
        )

        assert original_system != alchemical_system  # should be a copy

        nonbonded_forces = [
            force
            for force in alchemical_system.getForces()
            if isinstance(force, openmm.NonbondedForce)
        ]

        assert len(nonbonded_forces) == 1
        assert nonbonded_forces[0].getGlobalParameterName(0) == "lambda_electrostatics"
        assert is_close(
            nonbonded_forces[0].getParticleParameters(0)[2],
            0.0 * unit.kilojoule_per_mole,
        )

        custom_nonbonded_forces = [
            force
            for force in alchemical_system.getForces()
            if isinstance(force, openmm.CustomNonbondedForce)
        ]
        assert len(custom_nonbonded_forces) == 2
