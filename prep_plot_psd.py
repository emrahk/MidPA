# coding: utf-8
# python pp.py build_ext --inplace
# This is a code to fft mdot
#
# LOGS
#
# Jun 24 2025
# Cleaning unnecessary function
#

import os, sys, gc

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


def prep_powspec(time,signal, rebin=None, trange=None):
    """
    Prepare the power spectrum from a time series signal.
    
    Parameters:
    - time: array of time values
    - signal: array of signal values
    
    Optional Parameters:
    - rebin: if specified, the signal will be rebinned to this number of points
    - trange: if specified, only the signal within this time range will be considered

    Returns:
    - positive_freqs: frequencies corresponding to the positive half of the FFT
    - positive_power: power spectrum corresponding to the positive frequencies
    """

    if trange is not None:
        # Filter the signal based on the specified time range
        mask = (time >= trange[0]) & (time <= trange[1])
        time = time[mask]
        signal = signal[mask]
    
    # Number of samples
    N = len(signal)
    
    # Sampling interval
    T = time[1] - time[0]
    
    # Compute FFT
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, T)
    
    # Compute power spectrum (magnitude squared of FFT)
    power_spectrum = np.abs(fft_result)**2
    
    # Only keep the positive frequencies
    positive_freqs = freqs[:N//2]
    positive_power = power_spectrum[:N//2]


    if rebin is not None:
        # Rebin the power spectrum if rebin is specified
        nbins = len(positive_freqs) // rebin
        if nbins > 0:
            positive_freqs = positive_freqs[:nbins * rebin].reshape(-1, rebin).mean(axis=1)
            positive_power = positive_power[:nbins * rebin].reshape(-1, rebin).mean(axis=1)
        else:
            print("Rebinning factor is too large for the number of points in the signal.")
    
    return positive_freqs, positive_power

#fr, pow = prep_powspec(ntimes, mdot010)

#plot_variables(fr, pow, 
#                   x_scale='log', y_scale='log',
#                   x_label='Freq (c/Rg)', y_label='Power',
#                   x_range=[fr.min(),fr.max()], y_range=[pow.min()*0.8,pow.max()*1.2], 
#                   title='Power Spectrum')
