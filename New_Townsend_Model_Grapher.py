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


def New_Townsend_Model_Grapher(n0,alpha,title,xlimits,ylimits,legend,gridlines,xlabel,ylabel,discrete,continuous,exponential): #txt is true if dealing with a txt file, and false if dealing with a csv
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
    
    if discrete == True:
        xp = np.linspace(0,100,10000)  #index for graphing
        i = 0
            
        yp = np.linspace(0,100,10000)
        #testset = xp
    
        while i < len(xp):  #the function
            xpvalue = xp[i]
            seriesvalue = 0
        
            j = 0
        
            while j < (alpha*xpvalue - 1):  #everything in the series term
                seriesvalue = seriesvalue + (2**j)*alpha*(xpvalue-(j+1)/alpha)
                j = j + 1
        
            yp[i] = n0*(1+alpha*xpvalue+seriesvalue)
            #testset[i] = seriesvalue
            i = i + 1
    
        #logyp = np.log(yp)
        #loglogyp = np.log(logyp)
        
        ax.plot(xp, yp, '.', color=colorlist[0],label='Discrete')
        #ax.plot(xp, testset, '.', color=colorlist[0],label='My Model')
        
        """
        
        xp3 = np.linspace(0,100,100000)  #index for graphing
        i = 0
            
        yp3 = np.linspace(0,100,100000)
        #testset = xp
        
        alpha = 2
    
        while i < len(xp):  #the function
            xp3value = xp3[i]
            seriesvalue = 0
        
            j = 0
        
            while j < (alpha*xp3value - 1):  #everything in the series term
                seriesvalue = seriesvalue + (2**j)*alpha*(xp3value-(j+1)/alpha)
                j = j + 1
        
            yp3[i] = n0*(1+alpha*xp3value+seriesvalue)
            #testset[i] = seriesvalue
            i = i + 1
            
    
        #logyp = np.log(yp)
        #loglogyp = np.log(logyp)
        
        ax.plot(xp3, yp3, '.', color=colorlist[3],label='a = 2')
        #ax.plot(xp, testset, '.', color=colorlist[0],label='My Model')
        
       """ 
        
    if continuous == True:
        xp1 = np.linspace(0,100,10000)  #index for graphing
        
        i = 0
        
        yp1 = np.linspace(0,100,10000)
        yp2 = np.linspace(0,100,10000)
        
        while i < len(xp1):
            xp1value = xp1[i]
            intvalue = 0
            
            intvalueterm1 = (alpha*xp1value-1)*((np.log(2))**(-1))*((2**(alpha*xp1value-1))-1)
            intvalueterm2 = -((np.log(2))**(-1))*(alpha*xp1value-1)*2**(alpha*xp1value-1)
            intvalueterm3 = ((np.log(2))**(-2))*((2**(alpha*xp1value-1))-1)
            
            intvalueterm4 = ((np.log(2))**(-2))*(2**(alpha*xp1value-1))
                                                
            intvalue = intvalueterm1 + intvalueterm2 + intvalueterm3
            yp1[i] = n0*(1+alpha*xp1value+intvalue)
            yp2[i] = n0*(intvalueterm4)
            
            i = i + 1
        
        ax.plot(xp1, yp1, '.', color=colorlist[1],label='Full Model')
        ax.plot(xp1,yp2, '.', color = colorlist[2],label = 'Just Exponential Term')
        
        
        
    if exponential == True:
        
        xp2 = np.linspace(0,100,10000)  #index for graphing
        
        yp2 = n0*np.exp(alpha*xp2)
        
        ax.plot(xp2, yp2, '.', color=colorlist[2],label='Exponential')
    
    # make a plot 'ax', with markers and lines, color=red
    if xlimits != None:
        ax.set_xlim(xlimits[0],xlimits[1])   # set the range of the x-axis of plot 'ax'
        
    #ax.autoscale()
    
    if ylimits != None:
        ax.set_ylim(ylimits[0],ylimits[1])   # set the range of the y-axis
    #ax.set_xlabel('Potential, V')   # set the label of the x-axis
    #ax.set_ylabel('Current, mA/cm^2') # set the label of the y-axis
    if legend == True:
        ax.legend(loc='upper left')       # place the legend at the 'upper left'
    if gridlines == True:
        ax.xaxis.set_minor_locator(MultipleLocator(10))   # add minor ticks for the x-axis
        ax.yaxis.set_minor_locator(AutoMinorLocator())    # add minor ticks for the y-axis
        ax.xaxis.grid(True, which='both') # add grids to the x-axis for both major and minor ticks
    if title != None:
        ax.set_title(title)

    plt.show()   # display 'ax'
    #return(z[0]) #return slope of linear fit