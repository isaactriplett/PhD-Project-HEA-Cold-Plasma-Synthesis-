# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 02:46:03 2025

@author: isaac
"""

import pandas as pd
import numpy as np  # Numpy is a useful module for scientific computing (similar language to MATLAB)
import matplotlib.pyplot as plt  # import the module for plotting data
import math  # import the module containing mathematical functions
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)     # Configure tick location and format

"""Functions for obtaining and plotting mixing entropy of alloys based on atomic radii (modeling as spheres) and elemental composition"""

def Get_MixingEntropy(Atomiccomp):
    """Atomiccomp is a list, where the value of each index represents the coefficient of that element in the formula. The list goes as follows:
        [Li,Be,B,C,N,O,Na,Mg,Al,Si,P,S,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Ge,Se,Rb,Sr,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Te,Cs,Ba,La,Ce,
         Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu,Hf,Ta,W,Re,Os,Ir,Pt,Au,Tl,Pb,Po,Th,Pa,U]"""
        
    #radii
    
    rLi = 1.519
    
    Smix = 0
    
    return Smix