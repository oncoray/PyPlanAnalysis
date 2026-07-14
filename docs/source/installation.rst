Installation
============

PyPlanAnalysis uses a standard ``src``-layout package with a
``pyproject.toml`` build configuration.

There are two supported ways to install it, depending on whether your
environment has direct access to PyPI.

Standard install (PyPI access available)
-----------------------------------------

If your machine has normal internet/PyPI access

.. code-block:: bash

   git clone https://github.com/oncoray/PyPlanAnalysis.git
   cd PyPlanAnalysis
   pip install -e .

To also install development dependencies (``pytest``):

.. code-block:: bash

   pip install -e ".[dev]"

To also install documentation dependencies (``sphinx``, ``furo``, etc.):

.. code-block:: bash

   pip install -e ".[docs]"

Conda-based install (restricted/firewalled networks)
-------------------------------------------------------

If your network blocks direct access to PyPI (common on institutional or
clinical workstations), use ``Environment.yml`` to install all
dependencies via conda first, then install the package itself 

.. code-block:: bash

   git clone https://github.com/oncoray/PyPlanAnalysis.git
   cd PyPlanAnalysis
   conda env create -f Environment.yml
   conda activate PyPlan_env
   pip install -e . --no-deps --no-build-isolation

``--no-deps`` skips installing runtime dependencies via pip, since they
are already provided by the conda environment. ``--no-build-isolation``
avoids pip fetching build tools (``setuptools``, ``wheel``) into an
isolated build environment 

Requirements
------------

* Python >= 3.10
* Core dependencies (installed automatically via ``pip`` or provided
  via ``Environment.yml``): ``numpy``, ``pandas``, ``scipy``,
  ``matplotlib``, ``pydicom``, ``scikit-image``, ``openpyxl``,
  ``SimpleITK``.
* For the conda-based install: ``setuptools`` and ``wheel`` must also
  be listed in ``Environment.yml``.

The NTCP parameter workbook (``NTCPModels_params.xlsx``) is bundled
inside the package (``PyPlanAnalysis/data/``) and is located automatically
at runtime — no separate download or manual path configuration needed.