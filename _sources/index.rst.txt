PyPlanAnalysis
==============

**PyPlanAnalysis** is a Python package for DVH, LVH, and DLVH analysis of
proton therapy DICOM plans, with variable-RBE modelling and NTCP scoring.

It covers the full pipeline for one patient: auto-discovering and loading
the relevant DICOM files (CT, RTSTRUCT, RTPLAN, RTDOSE physical + LET),
extracting structure masks (binary or fractional/supersampled), computing
DVH/LVH/DLVH metrics under multiple RBE models, and scoring NTCP against a
bundled library of published toxicity models.

.. code-block:: python

   from PyPlanAnalysis import PatientPlan

   plan = PatientPlan.from_folder("path/to/patient_dicom_folder")
   results = plan.analyse(use_fractional=True)

   results.to_excel()
   results.plot_dvh()
   ntcp = results.CalcNTCP(NTCPConfig())

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   api/index
   ntcp_models

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
