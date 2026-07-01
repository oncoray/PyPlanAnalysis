# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:43:00 2026

@author: parrellgi
"""

from pathlib import Path

ROOT    = Path(__file__).parent.parent

SOURCE  = ROOT  /  "Source"
TEST    = ROOT  /  "Tests"

TEST_DATA       = TEST  / "test_data" 
TEST_DATA_INPUT =  "DATA" 
TEST_DATA_REFERENCE =  "Reference_DLVH" 

TEST_OUTPUT = TEST  / "test_output" 
TEST_PA_OUTPUT  =  Path("plan_analysis_outputs")
TEST_RESULTS    = Path("test_results")

UTILS  = ROOT / "Utils"