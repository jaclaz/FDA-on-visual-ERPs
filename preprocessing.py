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
from numpy import array
from mne.preprocessing import (ICA, create_eog_epochs)

home_path = os.path.abspath(os.getcwd())

def Preprocessing(paziente, home_path):
    
    categories = pd.read_csv(r"C:\Users\Asus\Downloads\things_concepts.tsv", sep='\t')
    
    filename = home_path+"\sub-"+str(paziente)+"\eeg\sub-"+str(paziente)+"_task-rsvp_events.csv"
    df_events = pd.read_csv(filename)
    
    filename = home_path+"\sub-"+str(paziente)+"\eeg\sub-"+str(paziente)+"_task-rsvp_eeg.vhdr"
    raw=mne.io.read_raw_brainvision(filename, preload=True)
    filtro=raw.copy().filter(0.1, 12, method='fir')
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
    
    #Rimuovo il canale di reference per evitare che venga conteggiato nelle epochs rimosse
    final_raw=reconst_raw.drop_channels('Reference')
    
    events, event_id = mne.events_from_annotations(final_raw)
    reject_criteria = dict(eeg=150e-6)    

    epochs = mne.Epochs(final_raw, events, tmin=-0.1, tmax=1)
    epochs.drop_bad(reject=reject_criteria)
    epochs.plot_drop_log()
    
    #Estraggo solo le epochs che registrano l'esperimento
    data=epochs['10001']
    
    #Estraggo le macro-categorie e le ordino per poterle associare facilmente
    #alle epochs corrispondenti
    
    BU_categories = categories['All Bottom-up Categories']
    obj_num=df_events['objectnumber']
    ord_cat=[]
    for ii in obj_num:
        if ii != -1:
            ord_cat.append(BU_categories[ii])

    new_events=data.events
    prova=obj_num.to_numpy()
    
    #imposto come codice evento i numeri associati ai concepts di THINGS
    #MEMO: devo costruire un nuovo dizionario per gli events
    
    for ii in np.arange(len(new_events[:])):
        new_events[ii][2]=prova[ii]
        
    #data.save('sub-'+str(paziente)+'_task-rsvp_epochs.fif', overwrite=True)
    
    return data, df_events, BU_categories, categories, ord_cat, new_events, obj_num, prova

for ii in range(41, 51):
    data, df_events, categories, tabellone, ord_cat, new_events, obj_num, prova = Preprocessing(ii, home_path)
    
    
    