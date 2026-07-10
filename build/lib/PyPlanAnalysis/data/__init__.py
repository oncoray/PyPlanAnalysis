"""
PyPlanAnalysis
============
DVH, LVH, and DLVH analysis for proton therapy DICOM plans, with
variable-RBE modelling and NTCP scoring. See the project README for
full usage details.
"""
 
from .Paths  import ROOT,SOURCE,TEST,TEST_DATA,TEST_DATA_INPUT,TEST_DATA_REFERENCE,TEST_OUTPUT,TEST_PA_OUTPUT,TEST_RESULTS,UTILS
 
__all__ = [
    "ROOT",
    "SOURCE","TEST","TEST_DATA","TEST_DATA_INPUT","TEST_DATA_REFERENCE","TEST_OUTPUT","TEST_PA_OUTPUT","TEST_RESULTS","UTILS"
]
 
__version__ = "1.0.0"
__author__  = "Giovanni Parrella"
 
