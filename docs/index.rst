============================================
ABsolute SOLVantion Free Energy Calculations
============================================

The `absolv` framework aims to offer a simple API for computing the change in free energy when transferring a solute
from one solvent to another, or to vacuum in the case of solvation free energy calculations.

It offers two routes to this end: standard equilibrium calculations and non-equilibrium switching type calculations,
where the latter will be the main focus of this framework.

.. toctree::
   :maxdepth: 2
   :hidden:

   Overview <self>
   installation

.. toctree::
   :hidden:
   :caption: User Guide

   user-guide/theory
   user-guide/transformations
   user-guide/reproducibility

.. toctree::
   :hidden:
   :caption: Examples

   examples/equilibrium
   examples/non-equilibrium

.. autosummary::
   :recursive:
   :caption: API
   :toctree: api
   :nosignatures:

    absolv
