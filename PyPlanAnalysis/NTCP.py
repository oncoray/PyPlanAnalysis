
"""
PyPlanAnalysis.NTCP.py
==============
computes different NTCP following details in /Utils/NTCPModels_params.xlsx. 
It works both with variable and fixed RBE models for proton therapy.

"""


# script to calculate NTCP based on DVH and clincial parameters
#_____________________________________________________________________________
import pandas as pd
import numpy as np
import scipy
from dataclasses import dataclass, field

@dataclass
class NTCPConfig:
    models    : list = field(default_factory=lambda:['Alopecia_G1_12m__1',
        'Alopecia_G1_12m__2',
        'Alopecia_G1_24m__1',
        'Alopecia_G1_24m__2',
        'Alopecia_G1_acute',
        'Alopecia_G2_acute',
        'Blindness_5y__Chiasma',
        'Blindness_5y__OpticNerve_ipsi',
        'Blindness_5y_OpticNerve_contra',
        'BrainNecrosis_5y__Brain-CTV',
        'BrainNecrosis_5y__BrainStem',
        'CataractRequiringIntervention_5y__Lens_ipsi',
        'CataractRequiringIntervention_5y__Lens_contra',
        'DelayedRecall_1_5y',
        'EndocrineDysfunction_late',
        'Erythema_G1_acute',
        'Erythema_G2_acute',
        'Fatigue_G1_24m',
        'Fatigue_G1_acute',
        'HearingImpairment_G1_12m__1__Cochlea_ipsi',
        'HearingImpairment_G1_12m__1__Cochlea_contra',
        'HearingImpairment_G1_12m__2__Cochlea_ipsi',
        'HearingImpairment_G1_12m__2__Cochlea_contra',
        'HearingImpairment_G1_24m__Cochlea_ipsi',
        'HearingImpairment_G1_24m__Cochlea_contra',
        'HearingLoss_late__Cochlea_ipsi',
        'HearingLoss_late__Cochlea_contra',
        'MemoryImpairment_G1_12m',
        'MemoryImpairment_G1_24m',
        'MemoryImpairment_G2_12m',
        'OcularToxicity_G2_acute__LacrimalGland_ipsi',
        'OcularToxicity_G2_acute__LacrimalGland_contra',
        'TemporalLobeInjury_5y__TemporalLobe_ipsi',
        'TemporalLobeInjury_5y__TemporalLobe_contra',
        'Tinnitus_G2_late__Cochlea_ipsi',
        'Tinnitus_G2_late__Cochlea_contra'])

# ____________________________________________________________________________
class NTCPModelBase():
    def __init__(self, model_name, df_models_path):
        self.model_name = model_name
        df_models = pd.read_excel(df_models_path)
        self.numberOfVariables = df_models["numberOfVariables"][df_models["model_name"]==self.model_name].values[0]
        self.parameterNames = []
        self.OAR_name =  df_models["OAR"][df_models["model_name"]==self.model_name].values[0]
        
        if "ipsi" in self.model_name:
            self.side = "ipsi"
        elif "contra" in self.model_name:
            self.side = "contra"
        else:
            self.side = None
            
        for i in range(1,self.numberOfVariables+1):
            self.parameterNames.append(df_models["parameterName_"+str(i)][df_models["model_name"]==model_name].values[0]) 
        
        self.impl_fn = npNan #eventuell auskommentieren
        if "Alopecia_G1_12m__1" in self.model_name:
            self.impl_fn = NTCP__Alopecia_G1_12m__1 
        elif "Alopecia_G1_12m__2" in self.model_name:
            self.impl_fn = NTCP__Alopecia_G1_12m__2 
        elif "Alopecia_G1_24m__1" in self.model_name:
            self.impl_fn = NTCP__Alopecia_G1_24m__1 
        elif "Alopecia_G1_24m__2" in self.model_name:
            self.impl_fn = NTCP__Alopecia_G1_24m__2 
        elif "Alopecia_G1_acute" in self.model_name:
            self.impl_fn = NTCP__Alopecia_G1_acute 
        elif "Alopecia_G2_acute" in self.model_name:
            self.impl_fn = NTCP__Alopecia_G2_acute 
        elif "Blindness_5y" in self.model_name:
            self.impl_fn = NTCP__Blindness_5y 
        elif "BrainNecrosis_5y" in self.model_name:
            self.impl_fn = NTCP__BrainNecrosis_5y 
        elif "CataractRequiringIntervention_5y" in self.model_name:
            self.impl_fn = NTCP__CataractRequiringIntervention_5y 
        elif "DelayedRecall_1_5y" in self.model_name:
            self.impl_fn = NTCP__DelayedRecall_1_5y 
        elif "EndocrineDysfunction_late" in self.model_name:
            self.impl_fn = NTCP__EndocrineDysfunction_late 
        elif "Erythema_G1_acute" in self.model_name:
            self.impl_fn = NTCP__Erythema_G1_acute 
        elif "Erythema_G2_acute" in self.model_name:
            self.impl_fn = NTCP__Erythema_G2_acute 
        elif "Fatigue_G1_24m" in self.model_name:
            self.impl_fn = NTCP__Fatigue_G1_24m 
        elif "Fatigue_G1_acute" in self.model_name:
            self.impl_fn = NTCP__Fatigue_G1_acute 
        elif "HearingImpairment_G1_12m__1" in self.model_name:
            self.impl_fn = NTCP__HearingImpairment_G1_12m__1 
        elif "HearingImpairment_G1_12m__2" in self.model_name:
            self.impl_fn = NTCP__HearingImpairment_G1_12m__2 
        elif "HearingImpairment_G1_24m" in self.model_name:
            self.impl_fn = NTCP__HearingImpairment_G1_24m 
        elif "HearingLoss_late" in self.model_name:
            self.impl_fn = NTCP__HearingLoss_late 
        elif "MemoryImpairment_G1_12m" in self.model_name:
            self.impl_fn = NTCP__MemoryImpairment_G1_12m 
        elif "MemoryImpairment_G1_24m" in self.model_name:
            self.impl_fn = NTCP__MemoryImpairment_G1_24m 
        elif "MemoryImpairment_G2_12m" in self.model_name:
            self.impl_fn = NTCP__MemoryImpairment_G2_12m 
        elif "OcularToxicity_G2_acute" in self.model_name:
            self.impl_fn = NTCP__OcularToxicity_G2_acute 
        elif "TemporalLobeInjury_5y" in self.model_name:
            self.impl_fn = NTCP__TemporalLobeInjury_5y 
        elif "Tinnitus_G2_late" in self.model_name:
            self.impl_fn = NTCP__Tinnitus_G2_late
      
    def define_side(self,vRBE_model, dfi_dvh,parameterName):
        rois = {}
        for s in dfi_dvh["ROI_Name"]:
            if self.OAR_name.lower() in s.lower():
                val = dfi_dvh[vRBE_model + '_' + parameterName][dfi_dvh["ROI_Name"] == s]
                rois[s] = val.values[0]
    
        # If only one (or zero) matching ROI, no side selection needed
        if len(rois) <= 1:
            return next(iter(rois), None)
    
        # Only apply ipsi/contra logic if the model name signals it
        
        if self.side == "ipsi" :
            return max(rois, key=lambda k: rois[k])
        elif self.side == "contra" :
            return min(rois, key=lambda k: rois[k])
        else:
            # Multiple ROIs but no side info in model name — fall back to first match
            return next(iter(rois))
        
    def compute_x(self,vRBE_model,dfi_dvh):
        x = []
        for i in range(0,self.numberOfVariables):
            try:
                if self.side:
                    roi = self.define_side(vRBE_model,dfi_dvh,self.parameterNames[i])
                    potential_x = [dfi_dvh[vRBE_model+'_'+self.parameterNames[i]][dfi_dvh["ROI_Name"]==roi]]
                    x.append(potential_x[0].values[0]) 
                else:
                    roi = None
                    for s in dfi_dvh["ROI_Name"]:
                        if self.OAR_name.lower() in s.lower():
                            roi = s
                    potential_x = [dfi_dvh[vRBE_model+'_'+self.parameterNames[i]][dfi_dvh["ROI_Name"]==roi]]
                    x.append(potential_x[0].values[0]) 
            except:
                x.append(np.nan)
        return x
    
    def compute_NTCP(self, vRBE_model,dfi_dvh):
        x = self.compute_x(vRBE_model,dfi_dvh)
        if not np.isnan(x).any():
            return np.round(self.impl_fn(x)*100,4)
            
def npNan(x):
    return np.nan

# ---------------------------------------------------------------------------------------------

# Alopecia grade ≥1_12 months after PBT
# __________________________________________________________________________
#    Late
#    Dutz et al. 2021
def NTCP__Alopecia_G1_12m__1(x, beta_0=-1.88, beta_1=0.15): #1.80 o 1.88??
#     # Skin V45Gy(RBE) in cm^(-3)
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



# Alopecia grade ≥1_12 months after PBT
# __________________________________________________________________________
#    Late
#    Dutz et al. 2021
def NTCP__Alopecia_G1_12m__2(x, beta_0=-6.38, beta_1=0.15):
#     # D2 == Skin D2% in Gy(RBE)^-(1)
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



# Alopecia grade ≥1_24 months after PBT
# __________________________________________________________________________
#    Late
#    Dutz et al. 2021
def NTCP__Alopecia_G1_24m__1(x, beta_0=-1.70, beta_1=0.048):
#     # Skin V30Gy(RBE) in cm^(-3)
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



# Alopecia grade ≥1_24 months after PBT
# __________________________________________________________________________
#    Late
#    Dutz et al. 2021
def NTCP__Alopecia_G1_24m__2(x, beta_0=-3.18, beta_1=0.068):
#     # D2 == Skin D2% in Gy(RBE)^-(1)
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



# Alopecia grade >=1 (CTCAE, Common Terminology Criteria for Adverse Events)
# __________________________________________________________________________
#    Acute
#    Dutz et al. 2019            
def NTCP__Alopecia_G1_acute(x, beta_0=-0.94, beta_1=0.10):
     # D2 == Skin D2%
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



# Alopecia grade >=2 (CTCAE, Common Terminology Criteria for Adverse Events)
# __________________________________________________________________________
#    Acute
#    Dutz et al. 2019
def NTCP__Alopecia_G2_acute(x, beta_0=-1.33, beta_1=0.081):
#     # D5 == Skin D5%
     return 1/(1+np.exp(-beta_0-beta_1*x[0]))



# Blindness
# __________________________________________________________________________
#    5 years post-RT
#    Burman et al. 1991
def NTCP__Blindness_5y(x, TD50=65.0, m=0.14):
    # chiasm and optic nerves gEUD, a = 4.0
    t = (x[0] - TD50)/(m*TD50)
    # equals 1/sqrt(2pi) integral_-inf^t exp(-x^2/2)dx
    return 0.5*(1+scipy.special.erf(t/np.sqrt(2)))



# Brain necrosis
# __________________________________________________________________________
#    5 years post-RT
#    Bender et al. 2012
def NTCP__BrainNecrosis_5y(x, D50=109.0, gamma=2.8):
    # Brain-CTV and brain stem Dmax (EQD2)
    if x[0] == 0:
        return 0
    else:
        return 1/(1+(D50/x[0])**(4*gamma))


    
# Cataract requiring intervention
# __________________________________________________________________________
#    5 years post-RT
#    Burman et al. 1991
def NTCP__CataractRequiringIntervention_5y(x, TD50=18.0, m=0.27):
    # Lenses gEUD
    t = (x[0] - TD50)/(m*TD50)
    # equals 1/sqrt(2pi) integral_-inf^t exp(-x^2/2)dx
    return 0.5*(1+scipy.special.erf(t/np.sqrt(2)))



# Delayed recall (on Wechsler Memory scale III Word Lists)
# __________________________________________________________________________
#    1.5 years post-RT
#    Gondi et al. 2012
def NTCP__DelayedRecall_1_5y(x, EQD_2_50=14.88, m=0.540):
#     # Bilateral hippocampi D40% (EQD2)
    t = (x[0] - EQD_2_50)/(m*EQD_2_50)
#     # equals 1/sqrt(2pi) integral_-inf^t exp(-x^2/2)dx
    return 0.5*(1+scipy.special.erf(t/np.sqrt(2)))



# Endocrine dysfunction (CTCAE, Common Terminology Criteria for Adverse Events)
# __________________________________________________________________________
#    At least 0.5 – 2 years post-RT
#    De Marzi et al. 2015
def NTCP__EndocrineDysfunction_late(x, TD50=60.5, gamma50=5.2):
#     # Pituitary gEUD, a = 6.4
    if x[0] == 0:
        return 0
    else:
        return 1/(1+(TD50/x[0])**(4*gamma50))

    
    
# Erythema grade ≥ 1
# __________________________________________________________________________
#    Acute
#    Dutz et al. 2019
def NTCP__Erythema_G1_acute(x, beta_0=1.00, beta_1=0.085):
#     # Skin V35Gy(RBE), absolute volume
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



# Erythema grade ≥ 2 (CTCAE, Common Terminology Criteria for Adverse Events)
# __________________________________________________________________________
#    Acute
#    Dutz et al. 2019
def NTCP__Erythema_G2_acute(x, beta_0=-1.54, beta_1=0.056):
#     # Skin V35Gy(RBE), absolute volume
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



# Fatigue grade ≥ 1_24 months after PBT
# __________________________________________________________________________
#    Late
#    Dutz et al. 2021
def NTCP__Fatigue_G1_24m(x, beta_0=-1.52, beta_1=0.021, beta_2=-1.16):
    # D2 == BrainStem D2% in Gy(RBE)^-(1), x[0]
    # CTx == 0: patient recieved no chemotherapy.x[1]
    # CTx == 1: patient recieved chemotherapy
    return 1/(1+np.exp(-beta_0-beta_1*x[0]-beta_2*x[1]))  



# Fatigue grade >=1 (CTCAE, Common Terminology Criteria for Adverse Events)
# __________________________________________________________________________
#    Acute
#    Dutz et al. 2019
def NTCP__Fatigue_G1_acute(x, beta_0=-0.90, beta_1=0.027, beta_2=1.28):
    # D2 == Brain-CTV(Gy) D2%, x[0]
    # female: gender = 1, x[1]
    # male:   gender = 0
    #return 1/(1+np.exp(-beta_0-beta_1*D2-beta_2*gender))
    return 1/(1+np.exp(-beta_0-beta_1*x[0]-beta_2*x[1]))



# Hearing impairment grade ≥1_12 months after PBT
# __________________________________________________________________________
#    Late
#    Dutz et al. 2021
def NTCP__HearingImpairment_G1_12m__1(x, beta_0=-3.03, beta_1=0.038):
    # Dmean == Cochlea ipsi Dmean in Gy(RBE)^-(1)
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



# Hearing impairment grade ≥1_12 months after PBT
# __________________________________________________________________________
#    Late
#    Dutz et al. 2021
def NTCP__HearingImpairment_G1_12m__2(x, beta_0=-7.02, beta_1=0.032, beta_2=0.072):
    # Dmean == Cochlea ipsi Dmean in Gy(RBE)^-(1), x[0]
    # Age == Age in years,x[1]
    #return 1/(1+np.exp(-beta_0-beta_1*Dmean-beta_2*Age))
    return 1/(1+np.exp(-beta_0-beta_1*x[0]-beta_2*x[1]))



# Hearing impairment grade ≥1_24 months after PBT
# __________________________________________________________________________
#    Late
#    Dutz et al. 2021
def NTCP__HearingImpairment_G1_24m(x, beta_0=-3.48, beta_1=0.050):
    # Dmean == Cochlea ipsi Dmean in Gy(RBE)^-(1)
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



# Hearing loss (CTCAE, Common Terminology Criteria for Adverse Events)
# __________________________________________________________________________
#    At least 0.5 – 2 years post-RT
#    De Marzi et al. 2015
def NTCP__HearingLoss_late(x, TD50=56.0, gamma50=2.9):
    # Cochlea gEUD, a = 1.2
    if x[0] == 0:
        return 0
    else:
        return 1/(1+(TD50/x[0])**(4*gamma50))
    

    
# Memory impairment grade ≥1_12 months after PBT
# __________________________________________________________________________
#    Late
#    Dutz et al. 2021
def NTCP__MemoryImpairment_G1_12m(x, beta_0=-2.32, beta_1=0.023):
#     # D2 == Hippocampi D2% in Gy(RBE)^-(1)
    return 1/(1+np.exp(-beta_0-beta_1*(x[0]/100)))



# Memory impairment grade ≥1_24 months after PBT
# __________________________________________________________________________
#    Late
#    Dutz et al. 2021
def NTCP__MemoryImpairment_G1_24m(x, beta_0=-1.77, beta_1=6.50):
    # Brain-CTV V35Gy(RBE) as fraction of the total volume
    return 1/(1+np.exp(-beta_0-beta_1*(x[0]/100)))



# Memory impairment grade ≥2_12 months after PBT
# __________________________________________________________________________
#    Late
#    Dutz et al. 2021
def NTCP__MemoryImpairment_G2_12m(x, beta_0=-3.42, beta_1=5.02):
    # Brain-CTV V25Gy(RBE) as fraction of the total volume
    return 1/(1+np.exp(-beta_0-beta_1*(x[0]/100)))



# Ocular toxicity grade ≥ 2 (RTOG, Radiation Therapy Oncology Group)
# __________________________________________________________________________
#    Acute
#    Batth et al. 2013
def NTCP__OcularToxicity_G2_acute(x, beta_0=-5.174, beta_1=0.205):
    # Dmax == Ipsilateral lacrimal gland Dmax
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



# Temporal lobe injury
# __________________________________________________________________________
#    5 years post-RT
#    Kong et al. 2016
def NTCP__TemporalLobeInjury_5y(x, beta_0=-18.61, beta_1=0.227):
#     # Dmax == Temporal lobe Dmax
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



# Tinnitus grade ≥ 2 (LENT-SOMA, late effects of normal tissues - subjective, objective, management)
# __________________________________________________________________________
#    1–2 years post-RT
#    Lee et al. 2015
def NTCP__Tinnitus_G2_late(x, TD50=46.52, m=0.35):
    #  Cochlea Dmean
    t = (x[0] - TD50)/(m*TD50)
    # equals 1/sqrt(2pi) integral_-inf^t exp(-x^2/2)dx
    return 0.5*(1+scipy.special.erf(t/np.sqrt(2)))
