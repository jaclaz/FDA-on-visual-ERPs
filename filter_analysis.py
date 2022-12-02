# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:39:45 2022

@author: Asus
"""

import numpy as np
import os
import pandas as pd
import mne
from mne.preprocessing import ICA

def filter_analysis(paziente):
    home_path = os.path.abspath(os.getcwd())
    filename = home_path+"\sub-0"+str(paziente)+"\eeg\sub-0"+str(paziente)+"_task-rsvp_eeg.vhdr"
    raw=mne.io.read_raw_brainvision(filename, preload=True)
    filt = mne.filter.create_filter(raw._data, sfreq=1000, l_freq=0.1, h_freq=12,
                             fir_design='firwin',phase='zero', verbose=True,
                             h_trans_bandwidth='auto', l_trans_bandwidth='auto',
                             filter_length='auto')
    mne.viz.plot_filter(filt, 1000, compensate=True)
    
filter_analysis(5)