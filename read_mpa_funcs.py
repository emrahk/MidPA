# coding: utf-8
# python pp.py build_ext --inplace
# This is a collection of codes to read RT dumps for the midplane
# approximation

import os, sys, gc
import shutil

import numpy as np
from distutils.core import setup
from setuptools import setup
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# add the current dir to the path
import inspect 	
#import matplotlib as mpl
#import matplotlib.pyplot as plt

this_script_full_path = inspect.stack()[0][1]
dirname = os.path.dirname(this_script_full_path)
sys.path.append(dirname) #do we need this? check later

def READ_RT_XTRACK(dir, dump):
    global N1, N3, t_read, Mdot_read, radius_read, tilt_read, prec_read, phi_read, h_proj_read, rho_proj_read, ug_proj_read, uukerr_proj_read, Normal_proj_read, source_proj_read
    global Mdot_phi_read,Rdot_phi_read, bary_phi_read, ph_phi_read, h_phi_read, rho_phi_read, ug_phi_read, source_phi_read
    global Te_phi_read, Ti_phi_read, Tr_phi_read, source_rad_phi_read
    f = open(dir + "/rt%d" % dump, "rb+")
    N1 = np.fromfile(f, dtype=np.int32, count=1, sep='')[0]
    N3 = np.fromfile(f, dtype=np.int32, count=1, sep='')[0]
    t_read = np.zeros((N1, N3), dtype=np.float32)
    Mdot_read = np.zeros((N1, N3), dtype=np.float32)
    array = np.fromfile(f, dtype=np.float32, count=N1 * N3 * 25, sep='').reshape((N1, N3, 25), order='C')
    array = np.swapaxes(array, 0, 2)
    array = np.swapaxes(array, 1, 2)
    t_read = array[0] #time
    Mdot_read = array[1] #mass accretion rate in code units
    tilt_read = array[2] #tilt in degrees
    prec_read = array[3] #precessiong angle in degrees
    radius_read = array[4] #radius of cell
    phi_read = array[5] #angle phi
    h_proj_read = array[6] #Midplane of disk
    rho_proj_read = array[7] #Integrated density in code units
    ug_proj_read = array[8] #Integrated internal energy in code units 
    uukerr_proj_read = array[9:13] #Kerr-4 velocity in spherical coordinates
    Normal_proj_read = array[13:16] #Normal vector (unnormalized) in spherical kerr-schild coordinates
    source_proj_read = array[16] #Integrated emmision dug/dt per surface area in code units
    Mdot_phi_read=array[17] #Mdot per radial and azimuthal bin
    Rdot_phi_read=array[18] #Rdot per radial and azimuthal bin
    bary_phi_read=array[19] #Cumulative barycentric radius per radial and azimuthal bin
    ph_phi_read=array[20] #What is the difference with phi_read?
    h_phi_read=array[21] #What is the difference with h_proj_read?
    rho_phi_read=array[22]
    ug_phi_read=array[23]
    source_phi_read=array[24]

    f.close()  

def calc_kmetr():
# Calculate kerr-metric for processed data coordinates
    global gcov_kerr, uukerr_proj_read, h_proj_read, radius_read
    global N1, N3, norm
    # Set constants
    a = 0.9375
    cth = np.cos(h_proj_read)
    sth = np.sin(h_proj_read)
    s2 = sth * sth
    rho2 = radius_read * radius_read + a * a * cth * cth
    gcov_kerr = np.zeros((4, 4, N1,  N3), dtype=np.float32)
    gcov_kerr[0, 0] = (-1. + 2. * radius_read / rho2)
    gcov_kerr[0, 1] = (2. * radius_read / rho2)
    gcov_kerr[0, 3] = (-2. * a * radius_read * s2 / rho2)
    gcov_kerr[1, 0] = gcov_kerr[0, 1]
    gcov_kerr[1, 1] = (1. + 2. * radius_read / rho2)
    gcov_kerr[1, 3] = (-a * s2 * (1. + 2. * radius_read / rho2))
    gcov_kerr[2, 2] = rho2
    gcov_kerr[3, 0] = gcov_kerr[0, 3]
    gcov_kerr[3, 1] = gcov_kerr[1, 3]
    gcov_kerr[3, 3] = (s2 * (rho2 + a * a * s2 * (1. + 2. * radius_read / rho2)))

    norm=gcov_kerr[0,0]*uukerr_proj_read[0]*uukerr_proj_read[0]+gcov_kerr[1,1]*uukerr_proj_read[1]*uukerr_proj_read[1]+gcov_kerr[2,2]*uukerr_proj_read[2]*uukerr_proj_read[2]+gcov_kerr[3,3]*uukerr_proj_read[3]*uukerr_proj_read[3]+2.0*(gcov_kerr[0,1]*uukerr_proj_read[0]*uukerr_proj_read[1]+gcov_kerr[0,2]*uukerr_proj_read[0]*uukerr_proj_read[2]+gcov_kerr[0,3]*uukerr_proj_read[0]*uukerr_proj_read[3]+gcov_kerr[2,1]*uukerr_proj_read[2]*uukerr_proj_read[1]+gcov_kerr[3,1]*uukerr_proj_read[3]*uukerr_proj_read[1]+gcov_kerr[2,3]*uukerr_proj_read[2]*uukerr_proj_read[3])
    print(norm.min(),norm.max())

def calc_aux_disk():
    global Temp, tau_bf, tau_es, Tg, Te_phi_read, Ti_phi_read, sigma,length_scale, length_scale, GAMMA, ENERGY_DENSITY_SCALE, time_scale
    global ARAD, rho_scale, C_CGS, BOLTZ_CGS, MU_E, MH_CGS, ENERGY_DENSITY_SCALE
    # Set constaints
    MH_CGS = 1.673534e-24  # Mass hydrogen molecule
    Z_AB = 0.02
    Y_AB = 0.28
    X_AB = 0.70
    MU_I=(4.0/(4.0*X_AB+Y_AB))
    MU_E=(2.0/(1.0+X_AB))
    MU_G=(4.0/(6*X_AB+Y_AB+2.0))
    BOLTZ_CGS = 1.3806504e-16  # Boltzmanns constant
    C_CGS = 3 * 10 ** 10  # cm/s
    G_CGS = 6.67 * 10 ** (-8)  # cm^3/g/s
    Msun = 2 * 10 ** 33  # g
    ARAD=7.5657e-15
    
    # Set black hole mass and EOS law
    Mbh = 10* Msun  # g (assuming 10 Msun BH)
    GAMMA = 5.0 / 3.0

    # Calculate lenght and timescales
    length_scale = G_CGS * Mbh / C_CGS ** 2  # cm
    time_scale = length_scale / C_CGS  # s

    # Set desired Mdot compared to Eddington rate
    L_dot_edd = 1.3 * 10 ** 46 * Mbh / (10 ** 8 * Msun)  # g/s
    efficiency = 0.178  # Look up for a=0.9375 black hole
    M_dot_edd = (1.0 / efficiency) * L_dot_edd / C_CGS ** 2  # g/s
    Mdot_desired = 0.35 * M_dot_edd  # According to paper

    # Calculate mass and energy density scales
    rho_scale = (Mdot_desired) / (Mdot_read * (length_scale ** 3) / time_scale)
    ENERGY_DENSITY_SCALE = (rho_scale * C_CGS * C_CGS)
    
    # Calculate temperature
    dU_scaled = source_proj_read * rho_scale * C_CGS ** 2 * (length_scale) / time_scale  # erg/cm^2/s
    sigma = 5.67 * 10 ** (-5)  # erg/cm^2/s/K
    Temp = np.nan_to_num((dU_scaled / sigma) ** (0.25) + 0.001)  # Kelvin
    Temp = Temp.astype(np.float64)

    # Calculate both bound-free and scattering optical depth across disk   
    Tg = MU_E * MH_CGS * (GAMMA - 1.) * (ug_proj_read * ENERGY_DENSITY_SCALE) / (BOLTZ_CGS * rho_proj_read * rho_scale)
    tau_bf = 30.0*3.0 * pow(10., 25.) * Tg ** (-3.5) * (rho_proj_read * rho_scale) ** 2 * length_scale
    tau_es = 0.4 * rho_proj_read * rho_scale * length_scale

