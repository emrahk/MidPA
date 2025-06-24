# coding: utf-8
# python pp.py build_ext --inplace
# This is a compilation of functions to plot multi dump outputs
#
# LOGS
#
# Jun 24 2025
# interpolate function now excludes 0 values
#

import numpy as np
from scipy.interpolate import interp1d

import matplotlib as mpl
import matplotlib.pyplot as plt
import platform

def enable_backend():
    """Set an interactive backend if running in IPython terminal."""
    try:
        from IPython import get_ipython
        ipy = get_ipython()
        if ipy is not None:
            if 'terminal' in str(type(ipy)).lower():
                import matplotlib
                matplotlib.use('MacOSX')
    except Exception as e:
        print(f"Could not set backend: {e}")


def prep_plot_multi(var, r,
               rmin=None, rmax=None):
    """
    Prepare data for plotting by calculating the average of var in bins defined by r.
    
    Parameters:
        var : 3D array of variable values
        r   : 2D array of radius values
        
    Returns:
        var_avgr : 1D array of average variable values in each bin
    """
    # Flatten the arrays to 1D, possibly not useful
    r = r.flatten()
    #var = var.flatten()

    # Remove NaN values from var and corresponding r values (later...)
    #mask = ~np.isnan(var)
    #r = r[mask]
    #var = var[mask]


    if rmin is None:
        rmin=np.min(r)
    if rmax is None:
        rmax=np.max(r)  #going to large r may be problematic

    #get number of bins
    nbins=var.shape[0]
    var_avgr=np.zeros((nbins), dtype=np.float64)

    for i in range(nbins):
        vari=var[i,:,:].flatten()
        var_avgr[i]=vari[(r>=rmin) & (r<rmax)].mean()

    return var_avgr


def prep_plot_multi_oa(var, r,
                       rmin=None, rmax=None, lengthscale=1.):
    """
    Prepare data for plotting by calculating the average of var*area in bins defined by r.
    
    Parameters:
        var : 3D array of variable values
        r   : 2D array of radius values
        
    Returns:
        var_avgr : 1D array of average variable values in each bin
    """
    # Flatten the arrays to 1D, possibly not useful
    r = r.flatten()
    #var = var.flatten()

    # Remove NaN values from var and corresponding r values (later...)
    #mask = ~np.isnan(var)
    #r = r[mask]
    #var = var[mask]


    if rmin is None:
        rmin=np.min(r)
    if rmax is None:
        rmax=np.max(r)  #going to large r may be problematic

    #get number of bins
    nbins=var.shape[0]
    var_oa=np.zeros((nbins), dtype=np.float64)

    for i in range(nbins):
        vari=var[i,:,:].flatten()
        # Remove NaN values from vari
        mask = ~np.isnan(vari)
        vari = vari[mask]
        r_masked = r[mask]
        var_oa[i]=vari[(r_masked>=rmin) & (r_masked<rmax)].mean()*np.pi*(rmax**2-rmin**2)*lengthscale**2

    return var_oa


def interpolate_times(times, var, dt=50.):
    """
    Interpolates the variable var to match the 50rg/c times.
    
    Parameters:
        times : 1D array of time values
        var   : 1D array of variable values
        dt    : interpolation parameter, default 50
        
    Returns:
        var_interp : 1D array of interpolated variable values
        times_interp : 1D array of interpolated time values
    """
    # Find the time range
    times2 = times[times >= 0.]  #consider only positive times
    var2 = var[times >= 0.]
    tmin = times2.min() - (times2.min() % dt)
    tmax = times2.max() + (dt - (times2.max() % dt))
    times_interp = np.arange(tmin, tmax+dt, dt)

    interp_func = interp1d(times2, var2, kind='linear', fill_value='extrapolate')
    var_interp = interp_func(times_interp)
    return times_interp, var_interp


def plot_variables(x, y, ye=None, expng=False, filename='plot.png', 
                   x_scale='linear', y_scale='linear',
                   x_label='X_axis', y_label='Y_axis',
                   x_range=None, y_range=None, 
                   title='My Plot'):
    """
    Plots y vs x with options for axis scaling, labels, and ranges.

    Parameters:
        x, y        : Arrays or lists of data points
        ye          : Array or list of error bars for y (optional)
        x_scale     : 'linear' or 'log'
        y_scale     : 'linear' or 'log'
        x_label     : Label for x-axis
        y_label     : Label for y-axis
        x_range     : Tuple (xmin, xmax) or None
        y_range     : Tuple (ymin, ymax) or None
        title       : Title of the plot
    """
#    enable_backend()


    if (platform.system() == 'Darwin'):
        enable_backend()

    plt.figure(figsize=(8, 6))
    plt.plot(x, y)  #marker='o'

        #if errors are provided
    if ye is not None:
        plt.errorbar(x, y, yerr=ye, fmt='o', ecolor='red', capsize=1)
        
    plt.xscale(x_scale)
    plt.yscale(y_scale)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)

    plt.grid(False)

    if expng:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    else:
#        enable_backend()
        plt.show()

    plt.close()  # close figure to avoid memory issues in loops
    
#    plt.show()

#example

#ymdot=prep_plot_multi(Mdot_phi_read_all, radius_read_all, rmin=0., rmax=10.)
#ymdot=prep_plot_multi(Mdotv2_all, radius_read_all, rmin=0., rmax=100.)

#ntimes, nymdot=interpolate_times(times, ymdot)

#dbbpow=prep_plot_multi_oa(dUscaled_all, radius_read_all, rmin=0., rmax=100.)
#ntimes, ndbbpow = interpolate_times(times, dbbpow)

'''
tmin=ntimes.min()-5.
tmax=ntimes.max()+5.

if nymdot.max() > 0. :
    ymax=nymdot.max()*1.2
elif nymdot.max() <= 0. :
    ymax=nymdot.max()*0.8

if nymdot.min() > 0. :
    ymin=nymdot.min()*0.8
elif nymdot.min() <= 0. :
    ymin=nymdot.min()*1.2


plot_variables(ntimes, nymdot, expng=False, filename='v2r180_2870_0_10.png',
                   x_scale='linear', y_scale='linear',
                   x_label='Time (Rg/c)', y_label='Mdot (g/s ?)',
                   x_range=[tmin,tmax], y_range=[ymin,ymax], 
                   title='Mdot vs Time')

if ndbbpow.max() > 0. :
    ymax=ndbbpow.max()*1.2
elif ndbbpow.max() <= 0. :
    ymax=ndbbpow.max()*0.8

if ndbbpow.min() > 0. :
    ymin=ndbbpow.min()*0.8
elif ndbbpow.min() <= 0. :
    ymin=ndbbpow.min()*1.2


plot_variables(ntimes, ndbbpow, expng=False, filename='dbbpowr180_2870_0_10.png',
                   x_scale='linear', y_scale='linear',
                   x_label='Time (Rg/c)', y_label='P (ergs/s)',
                   x_range=[tmin,tmax], y_range=[ymin,ymax], 
                   title='BB power vs Time')
'''
