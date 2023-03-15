# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:25:07 2022

@author: Asus
"""

import numpy as np
import os
from scipy.stats import f_oneway
import skfda
import pandas as pd
import mne
from skfda.inference.anova import oneway_anova
import scipy
from numpy import array
from mne.preprocessing import (ICA, create_eog_epochs)
import matplotlib.pyplot as plt


def EEG_preprocessed(paziente, filtro):
    
    home_path = os.path.abspath(os.getcwd())
    filename = home_path+"\sub-0"+str(paziente)+"\eeg\sub-0"+str(paziente)+"_task-rsvp_events.csv"
    df_events = pd.read_csv(filename)
    
    filename = home_path+"\sub-0"+str(paziente)+"\eeg\sub-0"+str(paziente)+"_task-rsvp_eeg.vhdr"
    raw=mne.io.read_raw_brainvision(filename, preload=True)
    filtro=raw.copy().filter(0.1, 12, method=filtro)
    reference=filtro.copy().set_eeg_reference(ref_channels='average')
    resample=reference.copy().resample(sfreq=250)
    notch=resample.copy().notch_filter(freqs=10)
    preica=notch.copy().filter(l_freq=1., h_freq=None)
    
    bipolar_ref=mne.set_bipolar_reference(preica, 'Fp1', 'Fp2', ch_name='Reference', drop_refs=False)
    reconst_raw=bipolar_ref.copy()
    
    ica = ICA(n_components=15, max_iter='auto', random_state=97)
    ica.fit(bipolar_ref)
    eog_indices, eog_scores = ica.find_bads_eog(bipolar_ref, ch_name='Reference')
    ica.exclude = eog_indices
    ica.apply(reconst_raw)
    
    final_raw=reconst_raw.drop_channels('Reference')
    
    return final_raw

#for ii in range(41, 51):
eeg_iir = EEG_preprocessed(4, 'iir')
eeg_fir = EEG_preprocessed(4, 'fir')
channels = eeg_fir.ch_names

def Plot(eeg_iir, eeg_fir, ch, channels):
    
    plt.title("Patient n°: 4, Channel: " + channels[ch])
    plt.plot(eeg_fir.times[0:2000], eeg_fir._data[ch][0:2000], label='FIR',color='#006494')
    plt.plot(eeg_iir.times[0:2000], eeg_iir._data[ch][0:2000], label='IIR',color='#EB6E34')
    plt.legend(loc='lower left')
    plt.savefig('Plots/' + channels[ch] , dpi=160)
    plt.close()

for ch in range(0, 63):
    Plot(eeg_iir, eeg_fir, ch, channels)




    
    