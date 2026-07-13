``PyPlanAnalysis.NTCP``
=========================

NTCP scoring engine: :class:`~PyPlanAnalysis.NTCP.NTCPConfig` selects which
models to run, :class:`~PyPlanAnalysis.NTCP.NTCPModelBase` loads one
model's metadata from the bundled parameter workbook and dispatches to its
``NTCP__*`` implementation function. See :doc:`../ntcp_models` for a
curated overview table of every model.

.. autofunction:: PyPlanAnalysis.NTCP.default_ntcp_params_path

.. autoclass:: PyPlanAnalysis.NTCP.NTCPConfig
   :members:
   :undoc-members:

.. autoclass:: PyPlanAnalysis.NTCP.NTCPModelBase
   :members:
   :undoc-members:

Model implementation functions
--------------------------------

.. automodule:: PyPlanAnalysis.NTCP
   :members:
   :exclude-members: NTCPConfig, NTCPModelBase, default_ntcp_params_path
   :undoc-members:
   :show-inheritance:
