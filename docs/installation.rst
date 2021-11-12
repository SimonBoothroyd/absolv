Installation
============

Installing using conda
----------------------

The recommended way to install ``absolv`` is via the ``conda`` package manger:

.. code-block:: bash

    # The -c SimonBoothroyd is temporary until a full release is made
    conda install -c conda-forge -c SimonBoothroyd absolv

If you have access to the OpenEye toolkits (namely ``oechem``, ``oequacpac`` and ``oeomega``) we recommend installing
these also as these can speed up molecule parameterization and coordinate generation significantly.

Installing from source
----------------------

To install ``absolv`` from source begin by cloning the repository from `github
<https://github.com/SimonBoothroyd/absolv>`_:

.. code-block:: bash

    git clone https://github.com/SimonBoothroyd/absolv.git
    cd absolv

Create a custom conda environment which contains the required dependencies and activate it:

.. code-block:: bash

    conda env create --name absolv --file devtools/conda-envs/test-env.yaml
    conda activate absolv

Finally, install ``absolv`` itself:

.. code-block:: bash

    python setup.py develop
