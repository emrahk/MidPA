# coding: utf-8
# python pp.py build_ext --inplace
# This is a code to create a blackbody spectrum from the temperatures obtained from the
# Mid plane approximation runs. 


import numpy as np
import matplotlib.pyplot as plt
import read_mpa_funcs as rmf
import plot_multi_funcs as pmf
import os, sys

dirname = "/data3/atakans/T65TOR_RT/RT3"


def bbspe(T, freqs):
    """
    Calculate the blackbody spectrum for a given temperature and frequency array.
    
    Parameters:
    - T: Temperature in Kelvin
    - freqs: Array of frequencies in Hz
    Returns:
    - Array of spectral radiance in ergs/s/cm^2/Hz (/sr?)
    """
    h = 6.62607015e-27  # Planck's constant in erg*s
    c = 2.99792458e10   # Speed of light in cm/s
    k_B = 1.380649e-16  # Boltzmann constant in erg/K

    # Calculate the spectral radiance using the Planck's law formula
    spectral_radiance = (2 * h * freqs**3 / c**2) / (np.exp(h * freqs / (k_B * T)) - 1)
    
    return spectral_radiance

def bbphspe(T, E, K):
    """
    Calculate the photon blackbody spectrum for a given temperature and energy array.
    Parameters:
    - T: Temperature in Kelvin
    - E: Array of photon energies in keV
    - K: Normalization factor (assumed to be 1 for L=10^39 ergs/s and D=10 kpc)
    Returns:
    - Array of photon spectra in photons/s/cm^2/keV
    """

    k_B = 1.380649e-16  # Boltzmann constant in erg/K
    erg2keV=624150648.
    kT = k_B * T * erg2keV  # Convert energy to keV
    A = K * 8.0525 * (E**2 / (kT)**4)*(1./(np.exp(E / (kT)) - 1)) # photon spectrum at 10 kpc

    return A

def findnorm(duscaled):
    """
    Find the normalization factor for the blackbody spectrum based on the given dump in rg/c
    to match the overall luminosity to 10^39 ergs/s at 10 kpc for an untilted disk.
    
    Parameters:
    - dump: dump number
    
    Returns:
    - Normalization factor (float)
    """
    
    #dU_scaled = rmf.source_proj_read * rmf.rho_scale * rmf.C_CGS ** 2 * (rmf.length_scale) / rmf.time_scale  # erg/cm^2/s
    
    #change the dimension of dU_scaled to match the expected shape
    dUscaled_all=np.zeros((1,rmf.N1,rmf.N3), dtype=np.float32)
    dUscaled_all[0,:,:]=duscaled[:,:]
    #integrate source
    source_int = 0. 

    for i in range(rmf.N1-1):
        minr = rmf.radius_read[i,0]
        maxr = rmf.radius_read[i+1,0]
        
        source_ann=pmf.prep_plot_multi_oa(dUscaled_all, rmf.radius_read, rmin=minr, 
                                      rmax=maxr, lengthscale=rmf.length_scale) #this does not work
        source_int += source_ann
    # Calculate the normalization factor
    # if T=1keV, L=10^39 ergs/s, K=1 at 10 kpc
    N=source_int/1.0e39 # 10^39 ergs/s

    return N

def calctiltspec(dump, erange, nbins, norm=None, silent=False):
    """
    Calculate the tilted disk blackbody spectrum for a given dump.
    
    Parameters:
    - dump: dump number
    - erange: energy range in keV
    - nbins: number of bins for the energy range
    
    Returns:
    - Tuple of arrays (energies, spectrum)
    """
    # read dump file
    dirname = "/data3/atakans/T65TOR_RT/RT3"
    rmf.READ_RT_XTRACK(dirname, dump)
    rmf.calc_aux_disk()  # calculate auxiliary disk properties

    # find normalization factor
    #K = findnorm(rmf.dU_scaled)
    #print("Normalization factor K:", K)
    # calculate energies (think about log later)
    energies = np.linspace(erange[0], erange[1], nbins)

    # calculate spectrum

    spectrum = np.zeros((rmf.N1,rmf.N3,nbins), dtype=np.float64)

    tot_area = np.pi * (rmf.radius_read.max()**2 - rmf.radius_read.min()**2) * rmf.length_scale**2  # correct this later
    tot_luminosity = 0.
    avg_temp = np.nanmean(rmf.Temp)  # average temperature for debugging
    for i in range(rmf.N1-1):
        for j in range(rmf.N3):
            cell_area = np.pi * (rmf.radius_read[i+1,j]**2 - rmf.radius_read[i,j]**2) * rmf.length_scale**2 / 512.
            tnorm = (cell_area/tot_area)*np.cos(rmf.tilt_read[i,j]*np.pi/180.) #check cos 
            # check Temp is not nan
            if ~np.isnan(rmf.Temp[i,j]):
                spectrum[i,j,:] = bbphspe(rmf.Temp[i,j], energies, tnorm)
                #spectrum[i,j,:] = bbphspe(11604525.0, energies, tnorm)
                tot_luminosity += rmf.dU_scaled[i,j] * cell_area
    #spectrum is in photons/s/cm^2/keV, so when writing the pha, we need per channel
    K=1.
    if norm is not None:
        K=tot_luminosity/1.0e39
    spectrum *= K  # If set normalize the spectrum to match the luminosity
    den = energies[1]-energies[0]
    totspectrum = np.sum(spectrum, axis=(0, 1))  # sum over all cells, ask Matthew for rmin?
    erg2keV=624150648.
    if not silent:  # print the results
        print("Total flux:", totspectrum.sum()*den, "photons/s/cm^2")
        print("Total flux:", (totspectrum*energies).sum()*den/erg2keV, "ergs/s/cm^2" )

        #luminosity at 10 kpc
        print("Luminosity at 10 kpc:", (totspectrum*energies).sum()*den*4.*np.pi*(10*3.086e21)**2/erg2keV, "ergs/s" )
        print("Total luminosity:", tot_luminosity, "ergs/s")

    return energies, spectrum


def collectspe(rtdrange, erange, nbins):
    """
    Collect the spectra from all dumps in the given directory.
    
    Parameters:
    - rtdrange: Range of dumps to process (tuple or list with start and end)
    - erange: Energy range in keV
    - nbins: Number of bins for the energy range
    
    Returns:
    - Tuple of arrays (energies, spectrum, exposure time, ligh curves?)
    """
    # Get list of dump files

    total_spectrum = np.zeros((rmf.N1,rmf.N3,nbins), dtype=np.float64)
    misdump=0

    #prep for light curves, make it more sophisticated later, just 1-3 keV, 3-6 keV for now
    time_scale = 4.9407407407407394e-05  # seconds, could have been read as well
    lc = np.zeros((rtdrange[1]-rtdrange[0]+1, 3), dtype=np.float64)

    for di in range(rtdrange[0], rtdrange[1]+1):
        print(f"Processing dump {di}...")
        dirname = "/data3/atakans/T65TOR_RT/RT3"
        if os.path.exists(dirname + "/rt%d" % di):
            energies, spectrum = calctiltspec(di, erange, nbins, silent=True)
            total_spectrum += spectrum  # accumulate the spectrum
            lc[di - rtdrange[0], 0] = rmf.t_read[0,0] * time_scale  # time in seconds
            lc[di - rtdrange[0], 1] = (spectrum[:, :, (energies >= 1) & (energies < 3)].sum()) * (energies[1]-energies[0])  # 1-3 keV
            lc[di - rtdrange[0], 2] = (spectrum[:, :, (energies >= 3) & (energies < 6)].sum()) * (energies[1]-energies[0])

        else:
            print(f"Dump {di} does not exist, skipping...")
            misdump +=1
            continue
    # calculate the exposure time
    
    exptime = (rtdrange[1] - rtdrange[0] - misdump) * 50. * time_scale # each dump is app 50rg/c apart

    return energies, total_spectrum, lc, exptime
    

def printspectrum(energies, spectrum, exptime, filename):
    """
    Print the spectrum to a file.
    
    Parameters:
    - energies: Array of energies in keV
    - spectrum: Array of spectral values
    - exptime: Exposure time in seconds
    - filename: Output filename
    """
    #exptime = 1.  # exposure time in seconds, will be changed later
    effarea = 500.0  # effective area in cm^2, will be changed later
    den = energies[1]-energies[0]
    en0 = energies[0]
    with open(filename, 'w') as f:
#        f.write("# Energy (keV)  Spectrum (photons/s/cm^2/keV)\n")   
        for i in range(len(energies)):
            val1 = en0+(i*den)
            val2 = en0+((i+1)*den)
            val3 = spectrum[i]*den
            val4 = np.sqrt(spectrum[i]*den* exptime*effarea) / (exptime * effarea)  # error in photons/s/cm^2/keV
            f.write(f"{val1:.4f} {val2:.4f} {val3:.8f} {val4:.8f}\n")
    print(f"Spectrum written to {filename}")

def prepplotlc(lc):
    """
    Prepare the light curves for plotting.
    
    Parameters:
    - lc: Array of light curve data (time, band1, band2)
    """
    effarea = 500.0  # effective area in cm^2, will be changed later
    # interpolate to reasonable intervals = 0.0025
    times = lc[:,0].flatten()
    lcs1 = lc[:,1].flatten()
    lcs2 = lc[:,2].flatten()

    ntimes, nlcs1 = pmf.interpolate_times(times, lcs1, dt=0.0025)
    ntimes, nlcs2 = pmf.interpolate_times(times, lcs2, dt=0.0025)

    #get errors
    elcs1 = np.sqrt(nlcs1*0.0025*effarea) / (0.0025 * effarea)  # error in photons/s/cm^2
    elcs2 = np.sqrt(nlcs2*0.0025*effarea) / (0.0025 * effarea)  # error in photons/s/cm^2

    plot_variables(ntimes, nlcs1, ye=elcs1, expng=True, filename='softlc.png', 
                   x_scale='linear', y_scale='linear',
                   x_label='Time (s)', y_label='cts/s',
                   x_range=None, y_range=None, 
                   title='Soft Light Curve')
    
    plot_variables(ntimes, nlcs2, ye=elcs2, expng=True, filename='hardlc.png',
                   x_scale='linear', y_scale='linear',
                   x_label='Time (s)', y_label='cts/s',
                   x_range=None, y_range=None, 
                   title='Hard Light Curve')
    
    return ntimes, nlcs1, elcs1, nlcs2, elcs2

