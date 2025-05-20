# coding: utf-8
# python pp.py build_ext --inplace
# In[21]:
# from __future__ import division__future__ import division
# from IPython.display import display

import os, sys, gc

import numpy as np

# add the current dir to the path
import inspect 	

this_script_full_path = inspect.stack()[0][1]
dirname = os.path.dirname(this_script_full_path)
sys.path.append(dirname)

import matplotlib as mpl
import matplotlib.pyplot as plt

import read_mpa_funcs as rmf

r=np.zeros((1,rmf.N1,1,rmf.N3), dtype=np.float32)
h=np.zeros((1,rmf.N1,1,rmf.N3), dtype=np.float32)
ph=np.zeros((1,rmf.N1,1,rmf.N3), dtype=np.float32)
var=np.zeros((1,rmf.N1,1,rmf.N3), dtype=np.float32)
nb2d=1
nb=1
D=100
t=rmf.t_read[0,0]
notebook=1
bs1new=rmf.N1
bs3new=rmf.N3
bs2new=1
r[0,:,0,:]=rmf.radius_read
h[0,:,0,:]=rmf.h_phi_read
ph[0,:,0,:]=rmf.phi_read

from matplotlib.gridspec import GridSpec
from distutils.dir_util import copy_tree

# add amsmath to the preamble
mpl.rcParams['text.latex.preamble'] = r"\usepackage{amssymb} \usepackage{amsmath}"
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

rc('text', usetex=False)
font = {'size': 40}
rc('font', **font)
rc('xtick', labelsize=70)
rc('ytick', labelsize=70)
# rc('xlabel', **int(f)ont)
# rc('ylabel', **int(f)ont)

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cmr10'
mpl.rcParams['font.sans-serif'] = 'cmr10'
plt.rcParams['image.cmap'] = 'jet'
if mpl.get_backend() != "module://ipykernel.pylab.backend_inline":
    plt.switch_backend('agg')
	

# needed in Python 3 for the axes to use Computer Modern (cm) fonts

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
legend = {'fontsize': 40}
rc('legend', **legend)
axes = {'labelsize': 50}
rc('axes', **axes)

fontsize = 38
mytype = np.float32

from sympy.interactive import printing

printing.init_printing(use_latex=True)

# For ODE integration
from scipy.integrate import odeint
from scipy.interpolate import interp1d

np.seterr(divide='ignore')

def plc_cart_xy1(var, min, max, rmax, offset, transform, name, label, linvar=0):
    fig = plt.figure(figsize=(64, 32))

    X = np.multiply(r, np.sin(ph))
    Y = np.multiply(r, np.cos(ph))
    if(transform==1):
        var2 = transform_scalar(var)
        var2 = project_vertical(var2)
    else:
        var2=var
    plotmax = int(10*rmax * np.sqrt(2))

    ilim = len(r[0, :, 0, 0]) - 1
    for i in range(len(r[0, :, 0, 0])):
        if r[0, i, 0, 0] > np.sqrt(2.0)*plotmax:
            ilim = i
            break

    plt.subplot(1, 2, 1)
    if (linvar == 1):
        var2p=var2
    else:
        var2p = np.log10(var2)
    
    res = plc_new_xy(var2p[:, 0:ilim], levels=np.arange(min, max, (max-min)/100.0), cb=0, isfilled=1, xcoord=X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1,z=offset, xmax=rmax, ymax=rmax)
    plt.xlabel(r"$x / R_g$", fontsize=90)
    plt.ylabel(r"$y / R_g$", fontsize=90)
    plt.title(label, fontsize=90)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', reset=False, which='both', length=24, width=6)
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(res, cax=cax)

    plt.subplot(1, 2, 2)
    res = plc_new_xy(var2p[:, 0:ilim], levels=np.arange(min, max, (max-min)/100.0), cb=0, isfilled=1, xcoord=X[:, 0:ilim], ycoord=Y[:, 0:ilim], xy=1, z=offset, xmax=rmax * 2.5, ymax=rmax * 2.5)
    plt.xlabel(r"$x / R_g$", fontsize=90)
    #plt.ylabel(r"$y / R_g$", fontsize=60)
    plt.title(label, fontsize=90)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', reset=False, which='both', length=24, width=6)
    plt.gca().set_aspect(1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb=plt.colorbar(res, cax=cax)
    plt.savefig(name, dpi=30)
    if (notebook == 0):
        plt.close('all')

def plc_new_xy(myvar, xcoord=None, ycoord=None, ax=None, **kwargs):  # plc
    global r, bs2new, notebook #h,ph
    l = [None] * nb2d
    # xcoord = kwargs.pop('x1', None)
    # ycoord = kwargs.pop('x2', None)
    if (np.min(myvar) == np.max(myvar)):
        print("The quantity you are trying to plot is a constant = %g." % np.min(myvar))
        return
    cb = kwargs.pop('cb', False)
    nc = kwargs.pop('nc', 15)
    k = kwargs.pop('k', 0)
    mirrory = kwargs.pop('mirrory', 0)
    # cmap = kwargs.pop('cmap',cm.jet)
    isfilled = kwargs.pop('isfilled', False)
    xy = kwargs.pop('xy', 1)
    xmax = kwargs.pop('xmax', 10)
    ymax = kwargs.pop('ymax', 5)
    z = kwargs.pop('z', 0)
    if ax is None:
        ax = plt.gca()
    if (nb > 1):
        if isfilled:
            for i in range(0, nb):
                if block[n_ord[i], AMR_COORD2] == (nb2 * np.power(1 + REF_2, block[n_ord[i], AMR_LEVEL2])//2):
                    res = ax.contourf(xcoord[i, :, 0, :], ycoord[i, :, 0, :], myvar[i, :, 0, :], nc,extend='both', **kwargs)
        else:
            for i in range(0, nb):
                if block[n_ord[i], AMR_COORD2] == (nb2 * np.power(1 + REF_2, block[n_ord[i], AMR_LEVEL2])//2):
                    res = ax.contour(xcoord[i, :, 0, :], ycoord[i, :, 0, :], myvar[i, :, 0, :], nc,extend='both', **kwargs)
    else:
        if isfilled:
            res = ax.contourf(xcoord[0, :, int(bs2new // 2), :], ycoord[0, :, int(bs2new // 2), :],myvar[0, :, int(bs2new // 2), :], nc, extend='both', **kwargs)
        else:
            res = ax.contour(xcoord[0, :, int(bs2new // 2), :], ycoord[0, :, int(bs2new // 2), :], myvar[0, :, int(bs2new // 2), :],nc, extend='both', **kwargs)
    if (cb == True):  # use color bar
        plt.colorbar(res, ax=ax)
    if (xy == 1):
        plt.xlim(-xmax, xmax)
        plt.ylim(-ymax, ymax)
    return res


#HERE IS THE ACTUAL PLOTTING


var=np.zeros((1,bs1new,1,bs3new), dtype=np.float32)

PI=3.1415926536
var[0,:,0,:]=2.*PI*rmf.rho_proj_read[:,:]*rmf.uukerr_proj_read[1,:,:]*rmf.radius_read[:,:]

if var.max() > 0. :
    ymax=var.max()*1.1
elif var.max() <= 0. :
    ymax=var.max()*0.9

if var.min() > 0. :
    ymin=var.min()*0.9
elif var.min() <= 0. :
    ymin=var.min()*1.1

notebook=1
plc_cart_xy1(var, ymin, ymax, 40., 0,0, dirname+ "/Mdotv2r_xy%d.png" % D, r"$([g/s])$ at %d $R_g/c$" % t, linvar=1)

