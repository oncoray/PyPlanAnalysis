# proton_dvh

Python module for computing dose-volume, LET-volume, and dose-LET-volume histograms from proton therapy DICOM plans.

## Features

- **DVH** – Physical dose, fixed RBE×1.1, and variable RBE (McNamara, Wedenberg, Carabe)
- **LVH** – LETd-volume histograms per structure
- **DLVH** – 2-D dose-LET-volume histogram (colourmap)
- **Metrics** – Dmin/mean/max, Dx%, Vx, gEUD, EQD2, dose-weighted LETd, LET-thresholded DVH, D_Lx%, L_Dx%
- **Outputs** – CSV, Excel, multi-page PDF (DVH/LVH), per-structure PNG (DLVH)
- **Batch mode** – loop over a patient cohort, aggregate into one CSV

---

## Installation

```bash
git clone https://github.com/yourname/proton_dvh.git
cd proton_dvh
pip install -e .
```

Or install dependencies manually:
```bash
pip install -r requirements.txt
```

---

## Quick start

### Single patient

```python
from proton_dvh import PatientPlan, RBEConfig, MetricConfig

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
from proton_dvh import RBEConfig, MetricConfig, RadiobiologyConfig

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

### Batch over a cohort

```python
from pathlib import Path
import pandas as pd
from proton_dvh import PatientPlan

all_dfs = []
for folder in Path("dataset/").iterdir():
    plan    = PatientPlan.from_folder(folder)
    results = plan.analyse()
    results.save_all(f"output/{folder.name}/")
    all_dfs.append(results.metrics_df)

pd.concat(all_dfs).to_csv("cohort_metrics.csv", index=False)
```

---

## Expected folder structure

```
dataset/
├── Patient_01/
│   ├── RD_physicaldose.dcm
│   ├── RD_LETd.dcm
│   └── RS_structures.dcm
└── Patient_02/
    └── ...
```

`from_folder()` auto-discovers files by DICOM modality and tags.  
The LET file is identified by `DoseType == "LET"` or `"LET"` appearing in `DoseComment` or the filename.

---

## Output structure

```
output/Patient_01/
├── dvh_metrics.csv       # all metrics, one row per structure
├── dvh_metrics.xlsx      # same, as Excel
├── dvh_curves.pdf        # DVH curves per dose type + per-structure comparison
├── lvh_curves.pdf        # LVH curves
└── dlvh_2d/
    ├── CTV.png
    └── Brainstem.png
```

---

## Module layout

```
proton_dvh/
├── __init__.py      # public API
├── io.py            # DICOM loading, grid resampling, structure masks
├── rbe.py           # RBE models (fixed, McNamara, Wedenberg, Carabe)
├── metrics.py       # DVH/LVH metric computation, histogram helpers
├── plots.py         # matplotlib plotting
└── plan.py          # PatientPlan and AnalysisResults classes
```

---

## Dependencies

- Python ≥ 3.8
- numpy, scipy, matplotlib, pandas, pydicom, dicompyler-core, openpyxl

---

## License

MIT
