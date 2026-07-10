# PyPlanAnalysis

Python module for computing dose-volume, LET-volume, and dose-LET-volume histograms from proton therapy DICOM plans. It allows also NTCP calculation for brain irradiations.

## Features

- **DVH** – Physical dose, fixed RBE×1.1, and variable RBE (McNamara, Wedenberg, Carabe)
- **LVH** – LETd-volume histograms per structure
- **DLVH** – 2-D dose-LET-volume histogram (colormap)
- **Metrics** – Dmin/mean/max, Dx%, Vx, gEUD, EQD2, dose-averaged LET, LETd-thresholded DVH, D_Lx%, L_Dx%
- **Outputs** – CSV, Excel, multi-page PDF (DVH/LVH), per-structure PNG (DLVH), NTCP summaries over RBE models

---

## Installation

```bash
git clone https://github.com/oncoray/PyPlanAnalysis.git
cd PyPlanAnalysis
```
### For Devs: Create PyPlan_env and install in editable (-e)
```bash
conda env create -f environment.yml
conda activate PyPlan_env
pip install -e . --no-build-isolation --no-deps
```
### For Users: Create a environment with requirements, then install 

```bash
conda create -n YourEnv -c conda-forge --file requirements_conda.txt
conda activate YourEnv
pip install . --no-deps  # or pip install git+https://github.com/oncoray/PlanAnalysis.git --no-deps
```

---

## Quick start

### Single patient

```python
from PyPlanAnalysis import PatientPlan, RBEConfig, MetricConfig

plan = PatientPlan(
    patient_id = "Patient_01",
    dose_file  = "RD_physicaldose.dcm",
    let_file   = "RD_LETd.dcm",
    rtstruct   = "RS_structures.dcm",
)

results = plan.analyse(structures=["CTV", "Brainstem"])
results.save_all("output/Patient_01/")
```

### Auto-discover DICOM files from a folder

```python
plan = PatientPlan.from_folder("data/Patient_01/")
results = plan.analyse()
results.to_csv("metrics.csv")
results.to_excel("metrics.xlsx")
```

### Custom RBE and metric configuration

```python
from PyPlanAnalysis import RBEConfig, MetricConfig, RadiobiologyConfig

rbe_cfg = RBEConfig(models=["mcnamara"], fixed_rbe=1.1)

metric_cfg = MetricConfig(
    dx = [2, 50, 98],
    vx = [20, 40],
)

radiobio_cfg = RadiobiologyConfig(
    alpha_beta_map     = {"ctv": 10.0, "brainstem": 2.0},
    alpha_beta_default = 3.0,
)

results = plan.analyse(rbe_cfg=rbe_cfg,
                       metric_cfg=metric_cfg,
                       radiobio_cfg=radiobio_cfg)
```
---

## Module layout

```
PyPlanAnalysis/
├── __init__.py      
├── io.py            # DICOM loading, grid resampling, structure masks
├── rbe.py           # RBE models (fixed, McNamara, Wedenberg, Carabe)
├── metrics.py       # DVH/LVH metric computation, histogram helpers
├── plots.py         # matplotlib plotting
├── NTCP.py          # NTCP config and computation
└── plan.py          # PatientPlan and AnalysisResults classes
```

---

## Dependencies (details coming soon)

- Python ≥ 3.10
- numpy, scipy, matplotlib, pandas, pydicom, openpyxl

---

