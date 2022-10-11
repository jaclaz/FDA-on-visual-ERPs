# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 17:22:18 2022

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
#from skfda.ml.classification import LogisticRegression
from numpy import array
from mne.preprocessing import (ICA, create_eog_epochs)

home_path = os.path.abspath(os.getcwd())

def Preprocessing(paziente, home_path):
    filename = home_path+"\sub-0"+str(paziente)+"\eeg\sub-0"+str(paziente)+"_task-rsvp_eeg.vhdr"
    raw=mne.io.read_raw_brainvision(filename, preload=True)
    filtro=raw.copy().filter(0.1, 12, method='iir')
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
    
    #file_annot = home_path+"\sub-0"+str(paziente)+"\eeg\sub-0"+str(paziente)+"_task-rsvp_events.csv"
    #annot_from_file = mne.read_annotations(file_annot)
    
    events, event_id = mne.events_from_annotations(reconst_raw)
    reject_criteria = dict(eeg=150e-6)    
    
    epochs = mne.Epochs(reconst_raw, events, tmin=-0.1, tmax=1, reject=reject_criteria)
    epochs.drop_bad(reject=reject_criteria)
    epochs.plot_drop_log()
    
    epochs.save('sub-0'+str(paziente)+'_task-rsvp_epochs.fif', overwrite=True)
    
    return epochs

for ii in range(1, 51):
    epochs= Preprocessing(ii, home_path)
    

    
    
    