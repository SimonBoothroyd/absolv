Alchemical Transformations
==========================

The ``absolv`` framework supports alchemically transforming the electrostatic and vdW interactions within an OpenMM
``System`` object via the ``OpenMMAlchemicalFactory`` object.

Due to the huge flexibility of the OpenMM system, allowing completely custom electrostatic and and vdW forces to be
implemented, it would be impossible to support all possible combinations of non-bonded forces. As such, the framework
currently only systems that contain

* vdW + electrostatic interactions in a single built-in ``NonbondedForce`` object
* electrostatic interactions in a built-in ``NonbondedForce`` object, vdW interactions in a ``CustomNonbondedForce``
  object and vdW 1-4 interactions in a ``CustomBondForce``
* the above combinations sans any electrostatics interactions

are supported.

Electrostatics
--------------

The electrostatic interactions in the system are alchemically transformed by linearly scaling all partial charges on
particles in *solute molecules* by :math:`\lambda_{elec}`, corresponding to a ``"lambda_electrostatics"`` context
variable that will be added by the factory, such that

.. math::

    q^{sol}_i \left(\lambda_{elec}\right) = \lambda_{elec} \times q^{sol}_i

All *intramolecular* interactions will be switched off during the alchemical transformation. This is referred to as
'annihilating' the electrostatic interactions in other packages and some literature.

Because the charges are scaled directly, the energy contributions of the alchemically scaled electrostatic interactions
will be

.. math::

   U^E = \lambda_{elec} U^E_{sol-solv} + \lambda_{elec}^2 U^E_{sol-sol} + U^E_{solv-solv}

where :math:`U^E_{sol-sol}`, :math:`U^E_{sol-solv}` and :math:`U^E_{solv-solv}` are the **un-scaled** electrostatic
contributions to the energies of the solute-solute, solute-solvent and solvent-solvent interactions respectively.

vdW
---

Currently vdW interactions can only be transformed if they are stored in a standard ``NonbondedForce`` **OR** if they are
split between a ``CustomBondForce`` (*1-2*, *1-3*, and *1-4* interactions) and a ``CustomNonbondedForce`` (the remaining
*1-n* and intermolecular interactions).

The interactions will be transformed according to :math:`\lambda_{vdW}` which corresponds to a ``"lambda_sterics"``
context variable that will be added by the factory.

Only *intermolecular* vdW interactions will be alchemically scaled, while all *intramolecular*
interactions will be left un-scaled. This is is referred to as 'decoupling' the vdW interactions in other packages and
some literature.

Lennard--Jones
""""""""""""""

If the vdW interactions are stored in a standard ``NonbondedForce``, then the alchemical factory will split them
so that

* the ``NonbondedForce`` force retains all interactions between solvent particles

* all intermolecular alchemical (both solute-solvent and solute-solute) interactions are moved into a new
  ``CustomNonbondedForce``

* all solute *intramolecular* interactions are moved into a new ``CustomBondForce``

  .. note:: The intramolecular solute-solute interactions won't use any periodic boundary corrections such that the the
            decoupled state of the solute corresponds to the proper vacuum state without periodicity effects.

The ``CustomNonbondedForce`` will copy over all settings (including cut-off, long-range correction, etc) from the
original ``NonbondedForce``, but will replace the normal LJ energy expression with a **soft-core** version. By default,
this takes the form:

.. math::

    U^{vdW} \left( \lambda_{vdW} \right) = \lambda_{vdW} \times 4 \varepsilon \left[ \left( \dfrac{\sigma}{r_{eff}}\right)^{12} - \left( \dfrac{\sigma}{r_{eff}}\right)^{6} \right]

where

.. math::

    r_{eff} = \sigma \left( \dfrac{1}{2} \left(1 - \lambda_{vdW}\right) + \left( \dfrac{r}{\sigma} \right) ^ 6 \right) ^ \frac{1}{6}

Custom vdW Forms
""""""""""""""""

If the vdW interactions are split across a ``CustomNonbondedForce`` and a ``CustomBondForce`` then the alchemical
factory will further split them so that

* the original ``CustomNonbondedForce`` force will retain all interactions between solvent particles

* all solute *intramolecular* interactions are moved into a new ``CustomNonbondedForce``

  .. note:: The intramolecular solute-solute interactions won't use any periodic boundary corrections such that the the
            decoupled state of the solute corresponds to the proper vacuum state without periodicity effects.

* all *intermolecular* alchemical (both solute-solvent and solute-solute) interactions are moved into another new
  ``CustomNonbondedForce`` with a modified energy expression such that

  .. math::

     U^{vdW}_{sol-solv} \left( \lambda_{vdW} \right) = \lambda_{vdW} \times U^{vdW}_{sol-solv}
