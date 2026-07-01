from setuptools import setup, find_packages

setup(
    name            = "PlanAnalysis",
    version         = "1.0.0",
    author          = "Giovanni Parrella",
    description     = "DVH/LVH/DLVH analysis for proton therapy DICOM plans",
    packages        = find_packages(),
    python_requires = ">=3.8",
    install_requires = [
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "pandas>=2.0",
        "pydicom>=2.4",
        "openpyxl>=3.1",
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
