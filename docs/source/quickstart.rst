Quickstart
==========

Loading a plan
---------------

The easiest way to build a :class:`~PyPlanAnalysis.plan.PatientPlan` is to
point :meth:`~PyPlanAnalysis.plan.PatientPlan.from_folder` at a directory
containing the patient's DICOM files (CT series, RTSTRUCT, RTPLAN, RTDOSE
physical dose, RTDOSE LET). It auto-discovers and cross-links the files
via their DICOM reference tags:

.. code-block:: python

   from PyPlanAnalysis import PatientPlan

   plan = PatientPlan.from_folder("path/to/patient_dicom_folder")

If your files aren't in one flat, fully self-consistent folder, build the
plan manually instead:

.. code-block:: python

   plan = PatientPlan(
       patient_id     = "Patient01",
       plan_file      = "RTPLAN.dcm",
       dose_file      = "RTDOSE_physical.dcm",
       let_file       = "RTDOSE_LET.dcm",
       rtstruct       = "RTSTRUCT.dcm",
       CT_folder_path = "CT/",
       n_fractions    = 30,
   )

Running the analysis
----------------------

.. code-block:: python

   results = plan.analyse(
       structures     = ["CTV", "Brainstem", "Parotid_L"],  # or None for all
       use_fractional = True,   # sub-voxel accurate masks (recommended)
       supersample    = 4,
   )

:meth:`~PyPlanAnalysis.plan.PatientPlan.analyse` accepts
:class:`~PyPlanAnalysis.rbe.RBEConfig`,
:class:`~PyPlanAnalysis.metrics.MetricConfig`, and
:class:`~PyPlanAnalysis.metrics.RadiobiologyConfig` to customise which RBE
models, DVH/LVH metric points, and per-structure α/β / gEUD-a values are
used. Sensible defaults are applied when omitted.

Exporting results
-------------------

.. code-block:: python

   results.to_csv()          # <patient_id>/dvh_metrics.csv
   results.to_excel()        # <patient_id>/dvh_metrics.xlsx
   results.plot_dvh()        # <patient_id>/dvh_curves.pdf
   results.plot_lvh()        # <patient_id>/lvh_curves.pdf
   results.plot_dlvh()       # <patient_id>/dlvh_2d/*.png

   # or all of the above at once:
   results.save_all()

NTCP scoring
-------------

.. code-block:: python

   from PyPlanAnalysis import NTCPConfig

   ntcp_cfg = NTCPConfig(models=["BrainNecrosis_5y", "HearingLoss_late"])
   ntcp = results.CalcNTCP(ntcp_cfg)

``path_models_xls`` may be omitted: it defaults to the parameter workbook
bundled with the package. See :doc:`ntcp_models` for the full list of
available models.
