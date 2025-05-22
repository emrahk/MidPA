# coding: utf-8
# python pp.py build_ext --inplace
# This is a code to read multiple RT dumps and plot them for the midplane
# approximation. This is the version that fills the missing items with the previous values.

import os, sys, gc
import platform
import shutil

import numpy as np
from scipy.interpolate import interp1d

import matplotlib as mpl
import matplotlib.pyplot as plt

import read_mpa_funcs as rmf

PI=3.1415926536

rtdrange=[180,300] #range of dumps to read
n_filesi=rtdrange[1]-rtdrange[0]+1
n_files=n_filesi
dirname = "/data3/atakans/T65TOR_RT/RT3"

dump=rtdrange[0]
if os.path.exists(dirname + "/rt%d" % dump):

    rmf.READ_RT_XTRACK(dirname, rtdrange[0])
    print("Reading dump %d" % rtdrange[0])
    print(rmf.t_read[0,0])
    rmf.calc_aux_disk()

    N1=rmf.N1
    N3=rmf.N3

    times=np.zeros((n_files), dtype=np.float32)
    radius_read_all=np.zeros((N1,N3), dtype=np.float32)
    h_phi_read_all=np.zeros((N1,N3), dtype=np.float32)
    phi_read_all=np.zeros((N1,N3), dtype=np.float32)
    rho_proj_read_all=np.zeros((n_files,N1,N3), dtype=np.float32)
    rho_scale_all=np.zeros((n_files,N1,N3), dtype=np.float32)
    Temp_all=np.zeros((n_files,N1,N3), dtype=np.float32)
    Mdotv2_all=np.zeros((n_files,N1,N3), dtype=np.float32)

    radius_read_all=rmf.radius_read
    h_phi_read_all=rmf.h_phi_read
    phi_read_all=rmf.phi_read
    rho_proj_read_all[0,:,:]=rmf.rho_proj_read
    rho_scale_all[0,:,:]=rmf.rho_scale
    Temp_all[0,:,:]=rmf.Temp
    times[0]=rmf.t_read[0,0]
    Mdotv2_all[0,:,:]=2.*PI*rmf.rho_proj_read[:,:]*rmf.uukerr_proj_read[1,:,:]*rmf.radius_read[:,:]
    
else:
    print("First Dump %d must exist, exiting" % dump)
    sys.exit(1)

for i in range(n_files-1):
    dump=rtdrange[0]+1+i
    print("Reading dump %d" % dump)
    if os.path.exists(dirname + "/rt%d" % dump):
        rmf.READ_RT_XTRACK(dirname, dump)
        #calc_kmetr()  I do not need this for now
        rmf.calc_aux_disk()
        print(rmf.t_read[0,0])
        times[i+1]=rmf.t_read[0,0]
        rho_proj_read_all[i+1,:,:]=rmf.rho_proj_read
        rho_scale_all[i+1,:,:]=rmf.rho_scale
        Temp_all[i+1,:,:]=rmf.Temp
        Mdotv2_all[i+1,:,:]=2.*PI*rmf.rho_proj_read[:,:]*rmf.uukerr_proj_read[1,:,:]*rmf.radius_read[:,:]
    else:
        print("Dump %d does not exist, skipping" % dump)
#        times[i+1]=times[i]+50. #Check this later
#        rho_proj_read_all[i+1,:,:]=rho_proj_read_all[i,:,:]
#        rho_scale_all[i+1,:,:]=rho_scale_all[i,:,:]
#        Temp_all[i+1,:,:]=Temp_all[i,:,:]
#        Mdotv2_all[i+1,:,:]=Mdotv2_all[i,:,:]


#clean skipped dumps
rho_proj_read_all=rho_proj_read_all[times>0.,:,:]
rho_scale_all=rho_scale_all[times>0.,:,:]
Temp_all=Temp_all[times>0.,:,:]
Mdotv2_all=Mdotv2_all[times>0.,:,:]
times=times[times>0.]

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
               rbin=1.0, rmin=None, rmax=None):
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
    var_avgr=np.zeros((nbins), dtype=np.float32)

    for i in range(nbins):
        vari=var[i,:,:].flatten()
        var_avgr[i]=vari[(r>=rmin) & (r<rmax)].mean()

    return var_avgr


def interpolate_times(times, var):
    """
    Interpolates the variable var to match the 50rg/c times.
    
    Parameters:
        times : 1D array of time values
        var   : 1D array of variable values
        
    Returns:
        var_interp : 1D array of interpolated variable values
        times_interp : 1D array of interpolated time values
    """
    # Find the time range
    tmin = times.min() - (times.min() % 50.)
    tmax = times.max() + (50. - (times.max() % 50.))
    times_interp = np.arange(tmin, tmax+1, 50)

    interp_func = interp1d(times, var, kind='linear', fill_value='extrapolate')
    var_interp = interp_func(times_interp)
    return times_interp, var_interp


def plot_variables(x, y, expng=False, filename='plot.png', 
                   x_scale='linear', y_scale='linear',
                   x_label='X_axis', y_label='Y_axis',
                   x_range=None, y_range=None, 
                   title='My Plot'):
    """
    Plots y vs x with options for axis scaling, labels, and ranges.

    Parameters:
        x, y        : Arrays or lists of data points
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
    
    plt.show()



#ymdot=prep_plot_multi(Mdot_phi_read_all, radius_read_all, rmin=0., rmax=10.)
ymdot=prep_plot_multi(Mdotv2_all, radius_read_all, rmin=0., rmax=10.)

ntimes, nymdot=interpolate_times(times, ymdot)

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


plot_variables(ntimes, nymdot, expng=True, filename='dene.png',
                   x_scale='linear', y_scale='linear',
                   x_label='Time (Rg/c)', y_label='Mdot (g/s ?)',
                   x_range=[tmin,tmax], y_range=[ymin,ymax], 
                   title='Mdot vs Time')

