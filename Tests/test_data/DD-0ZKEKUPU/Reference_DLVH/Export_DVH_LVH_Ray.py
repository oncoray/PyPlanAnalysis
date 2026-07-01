import math
import csv
import os
import numpy as np
from os import chdir, path, mkdir
from connect import *
import System, os, sys, clr


def GetDVH(aDoseObject, aExaminationName, aRoiName, NumberOfFractions):
    n = 2000
    aRelativeVolumes = [max(0.0, (x + 1.0)) / n for x in range(-1, n)]
    aRelativeVolumes[-1] = 1.0
    V = [v * 100 for v in aRelativeVolumes]
    HasContours = case.PatientModel.StructureSets[aExaminationName].RoiGeometries[aRoiName].HasContours()
    if HasContours:
        RoiVolume = case.PatientModel.StructureSets[aExaminationName].RoiGeometries[aRoiName].GetRoiVolume()
        DoseAtRelativeVolumes = aDoseObject.GetDoseAtRelativeVolumes(RoiName=aRoiName, RelativeVolumes=aRelativeVolumes)
        D = [d / 100 * NumberOfFractions for d in DoseAtRelativeVolumes]
    else:
        RoiVolume = 'nan'
        D = ['nan' for x in range(n + 1)]
    return RoiVolume, D, V
    

def GetLVH(aDoseObject, aExaminationName, aRoiName):
    n = 2000
    aRelativeVolumes = [max(0.0, (x + 1.0)) / n for x in range(-1, n)]
    aRelativeVolumes[-1] = 1.0
    V = [v * 100 for v in aRelativeVolumes]
    HasContours = case.PatientModel.StructureSets[aExaminationName].RoiGeometries[aRoiName].HasContours()
    if HasContours:
        RoiVolume = case.PatientModel.StructureSets[aExaminationName].RoiGeometries[aRoiName].GetRoiVolume()
        DoseAtRelativeVolumes = aDoseObject.GetDoseAtRelativeVolumes(RoiName=aRoiName, RelativeVolumes=aRelativeVolumes)
        D = [d/ 100 for d in DoseAtRelativeVolumes]
    else:
        RoiVolume = 'nan'
        D = ['nan' for x in range(n + 1)]
    return RoiVolume, D, V
    
patient_db = get_current("PatientDB")
patient = get_current("Patient")
case = get_current("Case")

export_folder = r'\\sv-onc-fs1\Home$\PARRELLGI\Documents\MBRO\DVH_code_local\test\Test_case_update'

if os.path.isdir(export_folder):
    os.chdir(export_folder)
else:
    os.mkdir(export_folder)
    os.chdir(export_folder)

if not os.path.exists(patient.PatientID):
    os.mkdir(patient.PatientID)
os.chdir(patient.PatientID)
	
plan = get_current('Plan')  

NumberOfFractions = plan.BeamSets[0].FractionationPattern.NumberOfFractions

RoiList = [Roi.Name for Roi in case.PatientModel.RegionsOfInterest]
RoiList.insert(0, ' ')

patient.Save()

ExaminationName = plan.BeamSets[0].FractionDose.OnDensity.FromExamination.Name
FD = plan.BeamSets[0].FractionDose

with open('NOMINAL_DVH_{}.csv'.format(plan.Name), 'w') as f:
    writer = csv.writer(f, delimiter=';', lineterminator='\n')
    writer.writerow(RoiList)
    RoiVolumeList = ['Vol']
    DVHMatrix = []
    DMeanList = ['Dmean']
    for iRoi, RoiName in enumerate(RoiList):
        if iRoi > 0:
            RoiVolume, D, V = GetDVH(FD, ExaminationName, RoiName, NumberOfFractions)
            Dmean = FD.GetDoseStatistic(RoiName=RoiName, DoseType="Average") / 100 * NumberOfFractions
            if iRoi == 1:
                DVHMatrix = [V, D]
            else:
                DVHMatrix.append(D)
            RoiVolumeList.append(RoiVolume)
            DMeanList.append(Dmean)
    DVHMatrix = list(map(list, zip(*DVHMatrix)))
    writer.writerow(RoiVolumeList)
    writer.writerows(DVHMatrix)
    writer.writerow(DMeanList)
						    
LET = plan.BeamSets[0].FractionDose.PhysicalDose.DoseValues.PhysicalData.DoseAveragedLetData*10 #/10 if compensate *10 from MeV/cm to keV/um + /100 for cGy to Gy (*10/100 = /10)

beamset = plan.BeamSets[0]
NumberOfVoxels      = beamset.FractionDose.InDoseGrid.NrVoxels
VoxelSize   = beamset.FractionDose.InDoseGrid.VoxelSize
Corner   = beamset.FractionDose.InDoseGrid.Corner

color_table = case.CaseSettings.LetColorMap.ColorTable
color_key = list(color_table.keys())
colors = list(color_table.values())
t = System.Drawing.Color.FromArgb(0,0,  0,  0 )
max_value = max(LET.flatten())
iso_lines = [100.0 / max_value * t for t in [int(i) for i in np.linspace(0,20,10)]] #SET MANUALLY FOR COLOR MAP

distribution_name = "LET_LVHexport"
case.AddAuxiliaryDataInDoseGrid(ExaminationName= case.Examinations[0].Name,  AuxiliaryUnit="LETd[KeV/um]", #SET AUXILIARY UNIT NAME MANUALLY
                                         Name= distribution_name,
                                         DoseGridNumberOfVoxels={ 'x': NumberOfVoxels['x'], 'y': NumberOfVoxels['y'], 'z': NumberOfVoxels['z']}, 
                                         DoseGridCorner={ 'x': Corner['x'], 'y': Corner['y'], 'z': Corner['z'] }, ReferenceDoseDistribution=None, 
                                         DoseGridVoxelSize={ 'x': VoxelSize['x'], 'y': VoxelSize['y'], 'z': VoxelSize['z'] }, Values=LET,
                                         ColorTable=color_table)

patient.Save()
list_eval = case.TreatmentDelivery.FractionEvaluations[0].DoseOnExaminations[0].DoseEvaluations
for evals in list_eval:
	if evals.Name==distribution_name:
		LET_for_LVH = evals

evals.UpdateDoseGridStructures()
patient.Save()

with open('NOMINAL_LVH_.csv', 'w') as f:
    writer = csv.writer(f, delimiter=';', lineterminator='\n')
    writer.writerow(RoiList)
    RoiVolumeList = ['Vol']
    DVHMatrix = []
    DMeanList = ['Dmean']
    for iRoi, RoiName in enumerate(RoiList):
        if iRoi > 0:
            RoiVolume, D, V = GetLVH(LET_for_LVH, ExaminationName, RoiName)
            Dmean = LET_for_LVH.GetDoseStatistic(RoiName=RoiName, DoseType="Average")  
            if iRoi == 1:
                DVHMatrix = [V, D]
            else:
                DVHMatrix.append(D)
            RoiVolumeList.append(RoiVolume)
            DMeanList.append(Dmean)
    DVHMatrix = list(map(list, zip(*DVHMatrix)))
    writer.writerow(RoiVolumeList)
    writer.writerows(DVHMatrix)
    writer.writerow(DMeanList)