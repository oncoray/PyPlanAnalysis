<p align="center">
  <img src="docs/logo.svg" alt="PyPlanAnalysis logo" width="500">
</p>

# PyPlanAnalysis

Python module for computing dose-volume, LET-volume, and dose-LET-volume histograms from proton therapy DICOM plans. It allows also NTCP calculation for brain irradiations.

## Features

- **DVH** – Physical dose, fixed RBE×1.1, and variable RBE (McNamara, Wedenberg, Carabe)
- **LVH** – LETd-volume histograms per structure
- **DLVH** – 2-D dose-LET-volume histogram (colormap)
- **Metrics** – Dmin/mean/max, Dx%, Vx, gEUD, EQD2, dose-averaged LET, LETd-thresholded DVH, D_Lx%, L_Dx%
- **Outputs** – CSV, Excel, multi-page PDF (DVH/LVH), per-structure PNG (DLVH), NTCP summaries over RBE models

---

## Documentation
Full documentation [here.](https://oncoray.github.io/PyPlanAnalysis/)

## Installation

```bash
git clone https://github.com/oncoray/PyPlanAnalysis.git
cd PyPlanAnalysis
conda env create -f Environment.yml
conda activate PyPlan_env
```
### For Users:  

```bash
pip install . --no-build-isolation  --no-deps  
```

### For Devs: install in editable format (-e)
```bash

pip install -e . --no-build-isolation --no-deps
```

### Alternative: use requirements_conda.txt to setup your own environment
```bash
conda create -n YourEnv -c conda-forge --file requirements_conda.txt
conda activate YourEnv
pip install . --no-build-isolation  --no-deps  
```
---

## Single patient Example
Run the example code on test data to have an example:

```PowerShell
cd PyPlanAnalysis
python -m Examples.single_test_patient
```


## Quick start
### Input DICOM files 

#### Auto-discover from a folder
```python
plan = PatientPlan.from_folder("data/Patient_01/")
```
#### Manual input
```python
from PyPlanAnalysis import PatientPlan, RBEConfig, MetricConfig

plan = PatientPlan(
    patient_id = "Patient_01",
    dose_file  = "RD_physicaldose.dcm",
    let_file   = "RD_LETd.dcm",
    rtstruct   = "RS_structures.dcm",
)
```

### Custom RBE and metric configuration

```python
from PyPlanAnalysis import RBEConfig, MetricConfig, RadiobiologyConfig

rbe_cfg = RBEConfig(models=["mcnamara", "linear"])

metric_cfg = MetricConfig(
    dx = [2, 50, 98],
    vx = [20, 40],
)

radiobio_cfg = RadiobiologyConfig(
    alpha_beta_map     = {"ctv": 10.0, "brainstem": 2.0},
    alpha_beta_default = 2.0,
)

NTCP_config = NTCPConfig() # uses built-in defaults
```
### Run the Analysis
```
results = plan.analyse(structures=["CTV", "Brainstem"],
                       rbe_cfg=rbe_cfg,
                       metric_cfg=metric_cfg,
                       radiobio_cfg=radiobio_cfg)
```

### Save results 
```
results.to_excel("metrics.xlsx")
#or
results.save_all("output/Patient_01/")

#save DLVH plots
results.plot_dlvh(output_dir / "dlvh_2d")
```

### Compute and save NTCP 
```
NTCP_summary = results.CalcNTCP(NTCP_config)

NTCP_summary = pd.DataFrame.from_dict(NTCP_summary)
save_ntcp = output_dir /  "NTCP_metrics.xlsx"
NTCP_summary.to_excel(save_ntcp)
```
---

## Module layout

```
PyPlanAnalysis/
├── __init__.py      
├── data             # Folder containing NTCPModels_params.xlsx for default NTCP computation
├── io.py            # DICOM loading, grid resampling, structure masks
├── rbe.py           # RBE models (fixed, McNamara, Wedenberg, Carabe, Linear (phys + 0.1 LET))
├── metrics.py       # DVH/LVH metric computation, histogram helpers
├── plots.py         # matplotlib plotting
├── NTCP.py          # NTCP config and computation
└── plan.py          # PatientPlan and AnalysisResults classes
```

---

## Dependencies (details coming soon)

- Python ≥ 3.10
- numpy>=1.24
- scipy>=1.10
- matplotlib>=3.7
- pandas>=2.0
- pydicom>=2.4
- openpyxl>=3.1
- scikit-image
- simpleitk
- pip
- setuptools 
- wheel

---

