NTCP Models Reference
=======================

PyPlanAnalysis bundles 25 published NTCP (Normal Tissue Complication
Probability) models, mostly for brain / head-and-neck proton therapy
toxicity endpoints. Each is implemented as one ``NTCP__*`` function in
:mod:`PyPlanAnalysis.NTCP`, driven by metadata (organ-at-risk, DVH/LVH
metric inputs, fitted coefficients) stored in the bundled
``NTCPModels_params.xlsx`` workbook.

Selecting models
------------------

Pass the desired model names to :class:`~PyPlanAnalysis.NTCP.NTCPConfig`:

.. code-block:: python

   from PyPlanAnalysis import NTCPConfig

   cfg = NTCPConfig(models=["BrainNecrosis_5y", "HearingLoss_late"])
   ntcp = results.CalcNTCP(cfg)

The ``models`` list must use the workbook's ``model_name`` values (the
part of each function name after ``NTCP__``), e.g. ``"BrainNecrosis_5y"``
for :func:`~PyPlanAnalysis.NTCP.NTCP__BrainNecrosis_5y`.

Available models
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 33 12 15 18

   * - Function
     - Endpoint
     - Timing
     - Reference
     - Parameters
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__Alopecia_G1_12m__1`
     - Alopecia grade >=1, 12 months after PBT
     - Late
     - Dutz et al. 2021
     - ``beta_0, beta_1``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__Alopecia_G1_12m__2`
     - Alopecia grade ≥1_12 months after PBT
     - Late
     - Dutz et al. 2021
     - ``beta_0, beta_1``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__Alopecia_G1_24m__1`
     - Alopecia grade ≥1_24 months after PBT
     - Late
     - Dutz et al. 2021
     - ``beta_0, beta_1``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__Alopecia_G1_24m__2`
     - Alopecia grade ≥1_24 months after PBT
     - Late
     - Dutz et al. 2021
     - ``beta_0, beta_1``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__Alopecia_G1_acute`
     - Alopecia grade >=1 (CTCAE, Common Terminology Criteria for Adverse Events)
     - Acute
     - Dutz et al. 2019
     - ``beta_0, beta_1``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__Alopecia_G2_acute`
     - Alopecia grade >=2 (CTCAE, Common Terminology Criteria for Adverse Events)
     - Acute
     - Dutz et al. 2019
     - ``beta_0, beta_1``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__Blindness_5y`
     - Blindness
     - 5 years post-RT
     - Burman et al. 1991
     - ``TD50, m``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__BrainNecrosis_5y`
     - Brain necrosis
     - 5 years post-RT
     - Bender et al. 2012
     - ``D50, gamma``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__CataractRequiringIntervention_5y`
     - Cataract requiring intervention
     - 5 years post-RT
     - Burman et al. 1991
     - ``TD50, m``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__DelayedRecall_1_5y`
     - Delayed recall (on Wechsler Memory scale III Word Lists)
     - 1.5 years post-RT
     - Gondi et al. 2012
     - ``EQD_2_50, m``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__EndocrineDysfunction_late`
     - Endocrine dysfunction (CTCAE, Common Terminology Criteria for Adverse Events)
     - At least 0.5 – 2 years post-RT
     - De Marzi et al. 2015
     - ``TD50, gamma50``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__Erythema_G1_acute`
     - Erythema grade ≥ 1
     - Acute
     - Dutz et al. 2019
     - ``beta_0, beta_1``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__Erythema_G2_acute`
     - Erythema grade ≥ 2 (CTCAE, Common Terminology Criteria for Adverse Events)
     - Acute
     - Dutz et al. 2019
     - ``beta_0, beta_1``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__Fatigue_G1_24m`
     - Fatigue grade ≥ 1_24 months after PBT
     - Late
     - Dutz et al. 2021
     - ``beta_0, beta_1, beta_2``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__Fatigue_G1_acute`
     - Fatigue grade >=1 (CTCAE, Common Terminology Criteria for Adverse Events)
     - Acute
     - Dutz et al. 2019
     - ``beta_0, beta_1, beta_2``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__HearingImpairment_G1_12m__1`
     - Hearing impairment grade ≥1_12 months after PBT
     - Late
     - Dutz et al. 2021
     - ``beta_0, beta_1``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__HearingImpairment_G1_12m__2`
     - Hearing impairment grade ≥1_12 months after PBT
     - Late
     - Dutz et al. 2021
     - ``beta_0, beta_1, beta_2``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__HearingImpairment_G1_24m`
     - Hearing impairment grade ≥1_24 months after PBT
     - Late
     - Dutz et al. 2021
     - ``beta_0, beta_1``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__HearingLoss_late`
     - Hearing loss (CTCAE, Common Terminology Criteria for Adverse Events)
     - At least 0.5 – 2 years post-RT
     - De Marzi et al. 2015
     - ``TD50, gamma50``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__MemoryImpairment_G1_12m`
     - Memory impairment grade ≥1_12 months after PBT
     - Late
     - Dutz et al. 2021
     - ``beta_0, beta_1``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__MemoryImpairment_G1_24m`
     - Memory impairment grade ≥1_24 months after PBT
     - Late
     - Dutz et al. 2021
     - ``beta_0, beta_1``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__MemoryImpairment_G2_12m`
     - Memory impairment grade ≥2_12 months after PBT
     - Late
     - Dutz et al. 2021
     - ``beta_0, beta_1``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__OcularToxicity_G2_acute`
     - Ocular toxicity grade ≥ 2 (RTOG, Radiation Therapy Oncology Group)
     - Acute
     - Batth et al. 2013
     - ``beta_0, beta_1``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__TemporalLobeInjury_5y`
     - Temporal lobe injury
     - 5 years post-RT
     - Kong et al. 2016
     - ``beta_0, beta_1``
   * - :func:`~PyPlanAnalysis.NTCP.NTCP__Tinnitus_G2_late`
     - Tinnitus grade ≥ 2 (LENT-SOMA, late effects of normal tissues - subjective, objective, management)
     - 1–2 years post-RT
     - Lee et al. 2015
     - ``TD50, m``
