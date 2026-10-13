# -*- coding: utf-8 -*-
"""
Created on 2016-2026

@author: Andres Christen, jac at cimat.mx
"""



from numpy import arange, linspace
from matplotlib.pylab import subplots

def PlotFrozenDist( frv, q=1e-6, N=200, linestyle='-', marker='o', color='k', add_grid=True,
                   ax=None, **kwargs):
    """Plots the instanced (frozen) univariate probability distribution frv, the range
       to plot is chosen from the q quantile to the 1-q quantile.  Automatically
       checks if dist is discrte or continous.  It is assumed that all discrete distributions
       are supported in the integers.  For continous dist the plot range is divided
       in N equal parts for plotting. q may be a size 2 iterable such that the plotting
       range is from the q[0] quantile to the q[1] quantile. The rest  of the arguments
       control plotting and are passed to the function plot.
       Example:

       from scipy.stats import norm
       PlotFrozenDist( norm( loc=10, scale=2) ) # plot N( 10, 2)

        returns the Axes
    """

    if (isinstance(q, float)):
        q0 = q
        q1 = 1.0-q
    else:
        q0 = q[0]
        q1 = q[1]
    
    if ax is None:
        fig, ax = subplots( figsize=(5,5))
    
    ### Check the type of distribution:
    continous = True
    try:
        frv.pdf(0)
    except:
        continous = False
    
    if (continous):
        x = linspace( frv.ppf(q0), frv.ppf(q1), N)
        y = frv.pdf(x)
        ax.plot(x, y, linestyle=linestyle, color=color, **kwargs)
        if add_grid:
            ax.grid(which='both', color='grey', linestyle='dotted', linewidth=0.5)
    else:
        x = arange( frv.ppf(q0), frv.ppf(q1), dtype=int)
        y = frv.pmf(x)
        ax.plot(x, y, marker, color=color, **kwargs)
        ax.vlines(x, 0, y,  colors=color, linestyles=linestyle)
    
    return ax

