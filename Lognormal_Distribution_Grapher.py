# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np  # Numpy is a useful module for scientific computing (similar language to MATLAB)
import matplotlib.pyplot as plt  # import the module for plotting data
import math  # import the module containing mathematical functions
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)     # Configure tick location and format


def Lognormal_Distribution_Grapher(sigma,mu,title,xlimits,ylimits,legend,gridlines,xlabel,ylabel): #txt is true if dealing with a txt file, and false if dealing with a csv
    #xlimits and ylimits set custom limits. Set to None if you want autoscaling.
    #legend is True if there is a legend.
    #gridlines is True if there are gridlines
    #xcolumn and ycolumn specify the columns which hold the relevant data

    fig = plt.figure(figsize=(8, 6))    # Create a graph 'fig' which has 4 inches in width and 6 inches in height.
    ax = fig.add_subplot(111)           # Create a subplot 'ax' in the figure 'fig'. 

    ax.set_ylabel(ylabel)   # set the label of the x-axis
    ax.set_xlabel(xlabel) # set the label of the y-axis
    
    X1 = []
    X2 = []
    X3 = []
    X4 = []
    X5 = []
    X6 = []
    X7 = []
    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    Y5 = []
    Y6 = []
    Y7 = []
    colorlist=['g','b','r','y','k','c','m']
    datalistX=[X1,X2,X3,X4,X5,X6,X7]
    datalistY=[Y1,Y2,Y3,Y4,Y5,Y6,Y7]
    
    
    i = 0
    
    while i < len(sigma):
        datalistX[i] = np.linspace(0,4,10000)  #index for graphing
        
        datalistY[i] = (1/(sigma[i]*datalistX[i]*np.sqrt(2*np.pi)))*np.exp(-((np.log(datalistX[i])-mu[i])**2)/(2*sigma[i]**2))
        
        ax.plot(datalistX[i], datalistY[i], '.', color=colorlist[i],label='$\sigma$ = ' + str(sigma[i]) + ', $\mu$ = ' + str(mu[i]))
        
        i = i+1
    
 
    
    # make a plot 'ax', with markers and lines, color=red
    if xlimits != None:
        ax.set_xlim(xlimits[0],xlimits[1])   # set the range of the x-axis of plot 'ax'
        
    #ax.autoscale()
    
    if ylimits != None:
        ax.set_ylim(ylimits[0],ylimits[1])   # set the range of the y-axis
    #ax.set_xlabel('Potential, V')   # set the label of the x-axis
    #ax.set_ylabel('Current, mA/cm^2') # set the label of the y-axis
    if legend == True:
        ax.legend(loc='upper right')       # place the legend at the 'upper left'
    if gridlines == True:
        ax.xaxis.set_minor_locator(MultipleLocator(10))   # add minor ticks for the x-axis
        ax.yaxis.set_minor_locator(AutoMinorLocator())    # add minor ticks for the y-axis
        ax.xaxis.grid(True, which='both') # add grids to the x-axis for both major and minor ticks
    if title != None:
        ax.set_title(title)

    plt.show()   # display 'ax'
    #return(z[0]) #return slope of linear fit