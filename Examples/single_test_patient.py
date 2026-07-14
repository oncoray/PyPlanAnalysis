"""
examples/single_patient.py
==========================
Analyse a single patient's proton plan end to end: load, configure,
run, export, and score NTCP. Edit the paths and settings below before
running.
"""



from PyPlanAnalysis import PatientPlan, RBEConfig, MetricConfig, RadiobiologyConfig, NTCPConfig
from Tests import Paths
import pandas as pd

# ------------------------------------------------------------------
# 1. Point to your DICOM files
# ------------------------------------------------------------------
Test_folder_name = "DD-0ZKEKUPU"
data_dir = Paths.TEST_DATA / Test_folder_name / Paths.TEST_DATA_INPUT
output_dir = Paths.TEST_OUTPUT  / Test_folder_name / Paths.TEST_PA_OUTPUT

# Auto-discover files from a folder:
plan = PatientPlan.from_folder( folder = data_dir, 
                                n_fractions = 30)

## Alternatively, set manually:
# plan = PatientPlan(
#     dose_file  = data_dir / "Physicaldose.dcm",
#     let_file   = data_dir / "LETd_keV_um.dcm",
#     rtstruct   = data_dir / "RTSTRUCT.dcm",
#     CT_folder_path    = data_dir / "CT",
#     n_fractions = 30
# )

#%%
# ------------------------------------------------------------------
# 3. Configure the analysis
# ------------------------------------------------------------------
rbe_cfg = RBEConfig(
    models    = ["mcnamara", "wedenberg", "carabe", "linear"],
    fixed_rbe = 1.1,
)


metric_cfg = MetricConfig(
    dx           = [2, 5, 50, 95, 98],   # Dx% points
    lx           = [2,50,98],            # Lx% points
    vx           = [25, 30, 35, 45, 50], # Vx thresholds [Gy(RBE)]
    LET_thr      = 3.0,                  # Dirty dose LET thr (Dose delviered through high LET particles - above LET_thr) [keV/um]
    compute_eqd2 = True,
    compute_geud = True,
    New_grid     = [1.5,1.5,1.5]         # mm
)

radiobio_cfg = RadiobiologyConfig(
    alpha_beta_map = {
         "Brain__minus__CTVunion_RBE_V01"       : 0.96,
         "Brainstem"       : 0.96,
         "Hippocampus"     : 2.0,
    },
    alpha_beta_default = 2.0,
    geud_a_map = {
        "OpticNerve": 4.0,
        "Chiasm"    : 4.0,
        "Lens"      : 3.33,
        "Cochlea"   : 1.2,
        "Pituitary" : 6.4
    },
    geud_a_default = 1.0,
)

NTCP_config = NTCPConfig() # uses built-in defaults

#or Select specific ROIs
NTCP_config = NTCPConfig(
    models=["HearingLoss_late__Cochlea_ipsi", "HearingLoss_late__Cochlea_contra"],
    roi_overrides={"HearingLoss_late__Cochlea_ipsi": "Cochlea_L"},  # explicit ROI, skips OAR/side logic
    ctv_name="CTV_boost",  # optional; omit to auto-pick first "CTV" match
)

# ------------------------------------------------------------------
# 4. Run the analysis
# ------------------------------------------------------------------
results = plan.analyse(
    structures   = None,#['BrainStem', 'Chiasm'],# or None for all
    rbe_cfg      = rbe_cfg,
    metric_cfg   = metric_cfg,
    radiobio_cfg = radiobio_cfg,
    resample_on_CT  = True,
    resample_on_custom_grid    = False,
    use_fractional =  False, 
    supersample  = 4
)

# ------------------------------------------------------------------
# 5. Save outputs
# ------------------------------------------------------------------

# Option A: save everything at once
results.save_all(
    output_dir = output_dir,
    csv_name   = "dvh_metrics.csv",
    excel_name = "dvh_metrics.xlsx",
)

results.plot_dlvh(output_dir / "dlvh_2d")


# ------------------------------------------------------------------
# Compute NTCP models on matching rois name
# ------------------------------------------------------------------

NTCP_summary = results.CalcNTCP(NTCP_config,Paths.UTILS / "NTCPModels_params.xlsx")
NTCP_summary = pd.DataFrame.from_dict(NTCP_summary)
save_ntcp = output_dir /  "NTCP_metrics.xlsx"


NTCP_summary.to_excel(save_ntcp)
print (f"Saved NTCP metrics at '{save_ntcp}'")
