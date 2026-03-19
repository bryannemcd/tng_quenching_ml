"""
Contains function for plotting density contours

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def density_contour(xdat,ydat, axis, binsize =100, **kwargs):
    """
    Plots density contours based on the provided x and y data.

    Parameters:
    xdat (array-like): The x data points.
    ydat (array-like): The y data points.
    axis (matplotlib.axes.Axes): The axes to plot on.
    binsize (int, optional): The number of bins to use for meshgrid. Default is 100.
    Other keyword arguments are passed to the contour plot.

    Returns:
    None
    """
    binsize=100
    deltaX=(max(xdat)-min(xdat))/binsize
    deltaY=(max(ydat)-min(ydat))/binsize

    xmin=min(xdat)-deltaX
    xmax=max(xdat)+deltaX
    ymin=min(ydat)-deltaY
    ymax=max(ydat)+deltaY

    #create a 100x100 grid of points
    xx,yy=np.mgrid[xmin:xmax:100j,ymin:ymax:100j]
    positions=np.vstack([xx.ravel(),yy.ravel()])
    
    #create a kernel density estimate of the 2D data
    values=np.vstack([xdat,ydat])
    kernel=st.gaussian_kde(values)

    #reshape the kernel density estimate to the grid
    #equivalent to the height values over which density contours are drawn
    Z=np.reshape(kernel(positions).T,xx.shape)

    axis.contour(xx,yy,Z, **kwargs)
    return()
