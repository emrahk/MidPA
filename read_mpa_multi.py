# coding: utf-8
# python pp.py build_ext --inplace
# This is a code to read multiple RT dumps and plot them for the midplane
# approximation. This is the version that interpolates the missing dumps
#
# LOGS
#
# 28 May 2025
# Adding functionality to handle source_proj (tilt not taken into account)
#
# 24 June 2025 cleaned up functions that are now called from plot_multi_funcs
#



import os, sys, gc
import platform
import shutil

import numpy as np
from scipy.interpolate import interp1d

import matplotlib as mpl
import matplotlib.pyplot as plt

import read_mpa_funcs as rmf

rtdrange=[180,2990] #range of dumps to read
n_filesi=rtdrange[1]-rtdrange[0]+1
n_files=0
dirname = "/data3/atakans/T65TOR_RT/RT3"

dump=rtdrange[0]
if os.path.exists(dirname + "/rt%d" % dump):

    rmf.READ_RT_XTRACK(dirname, rtdrange[0])
    print("Reading dump %d" % rtdrange[0])
    print(rmf.t_read[0,0])
    rmf.calc_aux_disk()

    N1=rmf.N1
    N3=rmf.N3
    dumps=np.zeros((n_filesi), dtype=np.uint16)
    times=np.zeros((n_filesi), dtype=np.float32)
    radius_read_all=np.zeros((N1,N3), dtype=np.float32)
    h_phi_read_all=np.zeros((N1,N3), dtype=np.float32)
    phi_read_all=np.zeros((N1,N3), dtype=np.float32)
    rho_proj_read_all=np.zeros((n_filesi,N1,N3), dtype=np.float32)
    rho_scale_all=np.zeros((n_filesi,N1,N3), dtype=np.float32)
    Temp_all=np.zeros((n_filesi,N1,N3), dtype=np.float32)
    Mdotv2_all=np.zeros((n_filesi,N1,N3), dtype=np.float32)
    dUscaled_all=np.zeros((n_filesi,N1,N3), dtype=np.float32)
    
    n_files=n_files+1
    dumps[0]=dump
    radius_read_all=rmf.radius_read
    h_phi_read_all=rmf.h_phi_read
    phi_read_all=rmf.phi_read
    rho_proj_read_all[0,:,:]=rmf.rho_proj_read
    rho_scale_all[0,:,:]=rmf.rho_scale
    Temp_all[0,:,:]=rmf.Temp
    times[0]=rmf.t_read[0,0]
    Mdotv2_all[0,:,:]=2.*np.pi*rmf.rho_proj_read[:,:]*rmf.uukerr_proj_read[1,:,:]*rmf.radius_read[:,:]
    #Energy density
    dU_scaled = rmf.source_proj_read * rmf.rho_scale * rmf.C_CGS ** 2 * (rmf.length_scale) / rmf.time_scale  # erg/cm^2/s
    dUscaled_all[0,:,:]=dU_scaled[:,:]
    
    
else:
    print("First Dump %d must exist, exiting" % dump)
    sys.exit(1)

for i in range(n_filesi-1):
    dump=rtdrange[0]+1+i
    print("Reading dump %d" % dump)
    if os.path.exists(dirname + "/rt%d" % dump):
        rmf.READ_RT_XTRACK(dirname, dump)
        #calc_kmetr()  I do not need this for now
        rmf.calc_aux_disk()
        print(rmf.t_read[0,0])
        n_files=n_files+1
        dumps[i+1]=dump
        times[i+1]=rmf.t_read[0,0]
        rho_proj_read_all[i+1,:,:]=rmf.rho_proj_read
        rho_scale_all[i+1,:,:]=rmf.rho_scale
        Temp_all[i+1,:,:]=rmf.Temp
        Mdotv2_all[i+1,:,:]=2.*np.pi*rmf.rho_proj_read[:,:]*rmf.uukerr_proj_read[1,:,:]*rmf.radius_read[:,:]
        dU_scaled = rmf.source_proj_read * rmf.rho_scale * rmf.C_CGS ** 2 * (rmf.length_scale) / rmf.time_scale  # erg/cm^2/s
        dUscaled_all[i+1,:,:]=dU_scaled[:,:]
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
dumps=dumps[times>0.]
dUscaled_all=dUscaled_all[times>0.,:,:]
times=times[times>0.]

'''
#ymdot=prep_plot_multi(Mdot_phi_read_all, radius_read_all, rmin=0., rmax=10.)
ymdot=prep_plot_multi(Mdotv2_all, radius_read_all, rmin=6.5, rmax=100.)

ntimes, nymdot=interpolate_times(times, ymdot)


dbbpow=prep_plot_multi_oa(dUscaled_all, radius_read_all, rmin=6.5, rmax=100.)
ntimes, ndbbpow = interpolate_times(times, dbbpow)

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


plot_variables(ntimes, nymdot, expng=True, filename='Mdot_180_2990_r6.5_r100.png',
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


plot_variables(ntimes, ndbbpow, expng=True, filename='dbbpowr180_2870_6.5_100.png',
                   x_scale='linear', y_scale='linear',
                   x_label='Time (Rg/c)', y_label='P (ergs/s)',
                   x_range=[tmin,tmax], y_range=[ymin,ymax], 
                   title='BB power vs Time')
'''
