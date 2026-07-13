
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
from importlib import resources


def default_ntcp_params_path():
    """
    Path to the NTCP model-parameter workbook bundled with the package
    (PyPlanAnalysis/data/NTCPModels_params.xlsx). Used automatically by
    NTCPModelBase when no explicit ``df_models_path`` is supplied, so
    installed users get working NTCP calculations out of the box.
    """
    return resources.files("PyPlanAnalysis.data").joinpath("NTCPModels_params.xlsx")

@dataclass
class NTCPConfig:
    """
    Selects which NTCP toxicity models to compute in
    ``AnalysisResults.CalcNTCP()``.

    Parameters
    ----------
    models : list of str
        Subset of the model names below (must match the ``model_name``
        column of the bundled ``NTCPModels_params.xlsx`` workbook).
        Default: all models listed below.

    Notes
    -----
    Each model name maps to one ``NTCP__*`` implementation function in
    this module (brain/head-and-neck late- and acute-toxicity endpoints
    from Dutz, De Marzi, Burman, Gondi, Kong, Lee, Bender and Batth).
    """
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
    """
    Loads one NTCP model's metadata from the parameter workbook and
    dispatches to its implementation function.

    Parameters
    ----------
    model_name : str
        Must match a value in the ``model_name`` column of the parameter
        workbook (see ``NTCPConfig.models`` for the full list).
    df_models_path : str or Path, optional
        Path to the NTCP parameter workbook (``.xlsx``). Default: None,
        which resolves to the workbook bundled with the package via
        ``default_ntcp_params_path()``.

    Attributes
    ----------
    OAR_name : str
        Organ-at-risk name this model applies to.
    numberOfVariables : int
        Number of covariates the model's implementation function expects.
    parameterNames : list of str
        DVH/LVH metric column-name suffixes to pull from the metrics
        DataFrame for each covariate, in order.
    side : {"ipsi", "contra", None}
        Laterality selection rule, inferred from ``model_name``.
    impl_fn : callable
        The ``NTCP__*`` function implementing this model's formula.
    """
    def __init__(self, model_name, df_models_path):
        self.model_name = model_name
        if df_models_path is None:
            df_models_path = default_ntcp_params_path()
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
        """
        Resolve which structure (ROI) to use when a model applies to a
        laterality-specific OAR (e.g. "Cochlea ipsi" vs "Cochlea contra")
        and the metrics DataFrame has more than one ROI matching
        ``self.OAR_name``.

        Parameters
        ----------
        vRBE_model : str
            Dose-type/RBE-model label prefix used in the metrics column
            names (e.g. "Phys", "RBE1.1", "mcnamara").
        dfi_dvh : pandas.DataFrame
            The patient's per-structure metrics table
            (``AnalysisResults.metrics_df``).
        parameterName : str
            Metric column-name suffix to compare across candidate ROIs.

        Returns
        -------
        str or None
            The chosen ``ROI_Name`` value, or None if no ROI matches.
        """
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
        """
        Build the covariate vector ``x`` this model's ``impl_fn`` expects,
        by pulling ``self.parameterNames`` columns for the matched ROI(s)
        out of the metrics DataFrame.

        Parameters
        ----------
        vRBE_model : str
            Dose-type/RBE-model label prefix (see ``define_side``).
        dfi_dvh : pandas.DataFrame
            Patient metrics table (``AnalysisResults.metrics_df``).

        Returns
        -------
        list of float
            One value per covariate, in ``self.parameterNames`` order.
            Entries are ``np.nan`` where the required ROI/metric could
            not be found.
        """
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
        """
        Compute this model's NTCP for one patient and one dose type.

        Parameters
        ----------
        vRBE_model : str
            Dose-type/RBE-model label prefix (see ``define_side``).
        dfi_dvh : pandas.DataFrame
            Patient metrics table (``AnalysisResults.metrics_df``).

        Returns
        -------
        float or None
            NTCP in percent (0-100), rounded to 4 decimals. Returns
            ``None`` if any required covariate is missing/NaN.
        """
        x = self.compute_x(vRBE_model,dfi_dvh)
        if not np.isnan(x).any():
            return np.round(self.impl_fn(x)*100,4)
            
def npNan(x):
    """Fallback implementation for an unrecognised model name; always returns NaN."""
    return np.nan

# ---------------------------------------------------------------------------------------------


def NTCP__Alopecia_G1_12m__1(x, beta_0=-1.88, beta_1=0.15): #1.80 o 1.88??
    """
    Alopecia grade >=1, 12 months after PBT
    Late
    Dutz et al. 2021

    Parameters
    ----------
    x : list
        Model covariates: x[0] = Skin V45Gy(RBE) in cm^-3.
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1].
    """
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



def NTCP__Alopecia_G1_12m__2(x, beta_0=-6.38, beta_1=0.15):
    """
    Alopecia grade ≥1_12 months after PBT
    Late
    Dutz et al. 2021

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1] 
    """
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))




def NTCP__Alopecia_G1_24m__1(x, beta_0=-1.70, beta_1=0.048):
    """
    Alopecia grade ≥1_24 months after PBT
    Late
    Dutz et al. 2021

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1] .
    """
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))




def NTCP__Alopecia_G1_24m__2(x, beta_0=-3.18, beta_1=0.068):
    """
    Alopecia grade ≥1_24 months after PBT
    Late
    Dutz et al. 2021

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1].
    """
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



      
def NTCP__Alopecia_G1_acute(x, beta_0=-0.94, beta_1=0.10):
    """
    Alopecia grade >=1 (CTCAE, Common Terminology Criteria for Adverse Events)
    Acute
    Dutz et al. 2019

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1].
    """
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))




def NTCP__Alopecia_G2_acute(x, beta_0=-1.33, beta_1=0.081):
    """
    Alopecia grade >=2 (CTCAE, Common Terminology Criteria for Adverse Events)
    Acute
    Dutz et al. 2019

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1]  .
    """
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



# Blindness
# __________________________________________________________________________
#    5 years post-RT
#    Burman et al. 1991
def NTCP__Blindness_5y(x, TD50=65.0, m=0.14):
    """
    Blindness
    Chiasm and optic nerves gEUD, a = 4.0
    5 years post-RT
    Burman et al. 1991

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    TD50 : float
        Fitted model coefficient (see reference above for origin/units).
    m : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1].
    """ 
    t = (x[0] - TD50)/(m*TD50)
    return 0.5*(1+scipy.special.erf(t/np.sqrt(2)))




def NTCP__BrainNecrosis_5y(x, D50=109.0, gamma=2.8):
    """
    Brain necrosis
    Brain-CTV and Brainstem Dmax (EQD2)
    5 years post-RT
    Bender et al. 2012

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    D50 : float
        Fitted model coefficient (see reference above for origin/units).
    gamma : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1].
    """
    if x[0] == 0:
        return 0
    else:
        return 1/(1+(D50/x[0])**(4*gamma))


    

def NTCP__CataractRequiringIntervention_5y(x, TD50=18.0, m=0.27):
    """
    Cataract requiring intervention
    Lenses gEUD
    5 years post-RT
    Burman et al. 1991

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    TD50 : float
        Fitted model coefficient (see reference above for origin/units).
    m : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1]  .
    """
    t = (x[0] - TD50)/(m*TD50)
    return 0.5*(1+scipy.special.erf(t/np.sqrt(2)))



def NTCP__DelayedRecall_1_5y(x, EQD_2_50=14.88, m=0.540):
    """
    Delayed recall (on Wechsler Memory scale III Word Lists)
    Bilateral hippocampi D40% (EQD2)
    1.5 years post-RT
    Gondi et al. 2012

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    EQD_2_50 : float
        Fitted model coefficient (see reference above for origin/units).
    m : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1].
    """
    t = (x[0] - EQD_2_50)/(m*EQD_2_50)
    return 0.5*(1+scipy.special.erf(t/np.sqrt(2)))




def NTCP__EndocrineDysfunction_late(x, TD50=60.5, gamma50=5.2):
    """
    Endocrine dysfunction (CTCAE, Common Terminology Criteria for Adverse Events)
    Pituitary gEUD, a = 6.4
    At least 0.5 – 2 years post-RT
    De Marzi et al. 2015

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    TD50 : float
        Fitted model coefficient (see reference above for origin/units).
    gamma50 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1]  .
    """  
    if x[0] == 0:
        return 0
    else:
        return 1/(1+(TD50/x[0])**(4*gamma50))

    
    

def NTCP__Erythema_G1_acute(x, beta_0=1.00, beta_1=0.085):
    """
    Erythema grade ≥ 1
    Skin V35Gy(RBE), absolute volume
    Acute
    Dutz et al. 2019

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1]  .
    """
#     
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))




def NTCP__Erythema_G2_acute(x, beta_0=-1.54, beta_1=0.056):
    """
    Erythema grade ≥ 2 (CTCAE, Common Terminology Criteria for Adverse Events)
    Skin V35Gy(RBE), absolute volume
    Acute
    Dutz et al. 2019

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1]  .
    """
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



def NTCP__Fatigue_G1_24m(x, beta_0=-1.52, beta_1=0.021, beta_2=-1.16):
    """
    Fatigue grade ≥ 1_24 months after PBT
    x[0] BrainStem D2% in Gy(RBE)^-(1)
    x[1] CTx == 0: patient recieved no chemotherapy
         CTx == 1: patient recieved chemotherapy
    Late
    Dutz et al. 2021

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_2 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1]  .
    """
    return 1/(1+np.exp(-beta_0-beta_1*x[0]-beta_2*x[1]))  




def NTCP__Fatigue_G1_acute(x, beta_0=-0.90, beta_1=0.027, beta_2=1.28):
    """
    Fatigue grade >=1 (CTCAE, Common Terminology Criteria for Adverse Events)
    x[0] Brain-CTV(Gy), D2%
    x[1] female: gender = 1
         male:   gender = 0
    Acute
    Dutz et al. 2019

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_2 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1]  .
    """
    return 1/(1+np.exp(-beta_0-beta_1*x[0]-beta_2*x[1]))




def NTCP__HearingImpairment_G1_12m__1(x, beta_0=-3.03, beta_1=0.038):
    """
    Hearing impairment grade ≥1_12 months after PBT
    Dmean == Cochlea ipsi Dmean in Gy(RBE)^-(1)
    Late
    Dutz et al. 2021

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1]  .
    """ 
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))




def NTCP__HearingImpairment_G1_12m__2(x, beta_0=-7.02, beta_1=0.032, beta_2=0.072):
    """
    Hearing impairment grade ≥1_12 months after PBT
    x[0] Dmean = Cochlea ipsi Dmean in Gy(RBE)^-(1)
    x[1] Age = Age in years
    Late
    Dutz et al. 2021

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_2 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1].
    """
    return 1/(1+np.exp(-beta_0-beta_1*x[0]-beta_2*x[1]))




def NTCP__HearingImpairment_G1_24m(x, beta_0=-3.48, beta_1=0.050):
    """
    Hearing impairment grade ≥1_24 months after PBT
    Cochlea ipsi Dmean in Gy(RBE)^-(1)
    Late
    Dutz et al. 2021

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1].
    """
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



def NTCP__HearingLoss_late(x, TD50=56.0, gamma50=2.9):
    """
    Hearing loss (CTCAE, Common Terminology Criteria for Adverse Events)
    Cochlea gEUD, a = 1.2
    At least 0.5 – 2 years post-RT
    De Marzi et al. 2015

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    TD50 : float
        Fitted model coefficient (see reference above for origin/units).
    gamma50 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1].
    """
    if x[0] == 0:
        return 0
    else:
        return 1/(1+(TD50/x[0])**(4*gamma50))
    

    
# Memory impairment grade ≥1_12 months after PBT
# __________________________________________________________________________
#    Late
#    Dutz et al. 2021
def NTCP__MemoryImpairment_G1_12m(x, beta_0=-2.32, beta_1=0.023):
    """
    Memory impairment grade ≥1_12 months after PBT
    Hippocampi D2% in Gy(RBE)^-(1)
    Late
    Dutz et al. 2021

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1] .
    """
    return 1/(1+np.exp(-beta_0-beta_1*(x[0]/100)))




def NTCP__MemoryImpairment_G1_24m(x, beta_0=-1.77, beta_1=6.50):
    """
    Memory impairment grade ≥1_24 months after PBT
    Brain-CTV V35Gy(RBE) as fraction of the total volume
    Late
    Dutz et al. 2021

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1] .
    """
    return 1/(1+np.exp(-beta_0-beta_1*(x[0]/100)))




def NTCP__MemoryImpairment_G2_12m(x, beta_0=-3.42, beta_1=5.02):
    """
    Memory impairment grade ≥2_12 months after PBT
    Brain-CTV V25Gy(RBE) as fraction of the total volume
    Late
    Dutz et al. 2021

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1] .
    """
    return 1/(1+np.exp(-beta_0-beta_1*(x[0]/100)))




def NTCP__OcularToxicity_G2_acute(x, beta_0=-5.174, beta_1=0.205):
    """
    Ocular toxicity grade ≥ 2 (RTOG, Radiation Therapy Oncology Group)
    Ipsilateral lacrimal gland Dmax
    Acute
    Batth et al. 2013

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1] .
    """
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))



# Temporal lobe injury
# __________________________________________________________________________
#    5 years post-RT
#    Kong et al. 2016
def NTCP__TemporalLobeInjury_5y(x, beta_0=-18.61, beta_1=0.227):
    """
    Temporal lobe injury
    Dmax = Temporal lobe Dmax
    5 years post-RT
    Kong et al. 2016

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    beta_0 : float
        Fitted model coefficient (see reference above for origin/units).
    beta_1 : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1] .
    """ 
    return 1/(1+np.exp(-beta_0-beta_1*x[0]))




def NTCP__Tinnitus_G2_late(x, TD50=46.52, m=0.35):
    """
    Tinnitus grade ≥ 2 (LENT-SOMA, late effects of normal tissues - subjective, objective, management)
    Cochlea Dmean
    1–2 years post-RT
    Lee et al. 2015

    Parameters
    ----------
    x : list
        Model covariates, in the order defined by the NTCP parameter
        workbook for this model (see NTCPModelBase.parameterNames).
    TD50 : float
        Fitted model coefficient (see reference above for origin/units).
    m : float
        Fitted model coefficient (see reference above for origin/units).

    Returns
    -------
    float
        NTCP probability in [0, 1] .
    """
    t = (x[0] - TD50)/(m*TD50)

    return 0.5*(1+scipy.special.erf(t/np.sqrt(2)))
