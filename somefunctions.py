# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 00:33:30 2025

@author: isaac
"""

import pandas as pd
import numpy as np  # Numpy is a useful module for scientific computing (similar language to MATLAB)
import matplotlib.pyplot as plt  # import the module for plotting data
import math  # import the module containing mathematical functions
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)     # Configure tick location and format

zeroto1 = np.linspace(0,1,1000)

cosines = 2**(0.5)*(np.cos(zeroto1*np.pi)+np.cos(3*zeroto1*np.pi))

hermite = (3**(0.5))*zeroto1 + 35*(zeroto1**3-3*zeroto1)/68

legendre = 12**(0.5)*(zeroto1-0.5)+7**(0.5)*(20*(zeroto1-0.5)**3-3*(zeroto1-0.5))

fig = plt.figure(figsize=(8, 6))    # Create a graph 'fig' which has 4 inches in width and 6 inches in height.
ax = fig.add_subplot(111)           # Create a subplot 'ax' in the figure 'fig'. 

ax.set_ylabel('p')   # set the label of the x-axis
ax.set_xlabel('u') # set the label of the y-axis

#ax.plot(zeroto1, cosines, '-', color='g',label='cosines')
#ax.plot(zeroto1, hermite, '-', color='g',label='Hermite')
ax.plot(zeroto1, legendre, '-', color='g',label='Legendre')
    

ax.legend(loc='upper right')       # place the legend at the 'upper left'      
