Installation
============

PyPlanAnalysis uses a standard ``src``-layout package with a
``pyproject.toml`` build configuration.

From a local clone
-------------------

.. code-block:: bash

   git clone https://github.com/<your-username>/PyPlanAnalysis.git
   cd PyPlanAnalysis
   pip install -e .

Requirements
------------

* Python >= 3.10
* Core dependencies (installed automatically): ``numpy``, ``pandas``,
  ``scipy``, ``matplotlib``, ``pydicom``, ``scikit-image``, ``openpyxl``,
  ``SimpleITK``.

The NTCP parameter workbook (``NTCPModels_params.xlsx``) is bundled
inside the package (``PyPlanAnalysis/data/``) and is located automatically
at runtime — no separate download or manual path configuration needed.

Development install
--------------------

To also install the test dependencies:

.. code-block:: bash

   pip install -e ".[dev]"
   pytest tests/
