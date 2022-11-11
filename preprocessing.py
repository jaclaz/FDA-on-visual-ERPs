# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 17:22:18 2022

@author: Asus
"""

import numpy as np
import os
import pandas as pd
import mne
from mne.preprocessing import ICA

def Preprocessing(filename):
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
    
    #Rimuovo il canale di reference per evitare che venga conteggiato nelle epochs rimosse
    final_raw=reconst_raw.drop_channels('Reference')
    
    return final_raw

def Start(paziente, home_path):
    
    categories = pd.read_csv(r"C:\Users\Asus\Downloads\things_concepts.tsv", sep='\t')
    
    filename = home_path+"\sub-0"+str(paziente)+"\eeg\sub-0"+str(paziente)+"_task-rsvp_events.csv"
    df_events = pd.read_csv(filename)
    
    filename = home_path+"\sub-0"+str(paziente)+"\eeg\sub-0"+str(paziente)+"_task-rsvp_eeg.vhdr"
    
    final_raw=Preprocessing(filename)
    events, event_id = mne.events_from_annotations(final_raw)    

    epochs = mne.Epochs(final_raw, events, tmin=-0.1, tmax=1)
    
    #Estraggo solo le epochs che registrano l'esperimento
    data=epochs['10001']
    
    #Estraggo le macro-categorie e le ordino per poterle associare facilmente
    #alle epochs corrispondenti
    BU_categories = categories['All Bottom-up Categories']
    obj_num=df_events['objectnumber']

    new_events=data.events
    prova=obj_num.to_numpy()
    
    #imposto come codice evento i numeri associati ai concepts di THINGS
    
    for ii in np.arange(len(new_events[:])):
        new_events[ii][2]=prova[ii]
        
    #nuovo dizionario per gli events con i concepts associati alle macro-categorie
    cat_dataframe = pd.read_csv(home_path+'\cat_dataframe.csv',index_col=0)
    new_dict=cat_dataframe.T.to_dict(orient='list')
    data.event_id=new_dict
    
    #elimino le epochs con ptp amplitude > 150 microvolt
    reject_criteria = dict(eeg=150e-6)
    data.drop_bad(reject=reject_criteria)
    data.plot_drop_log()
    
    #data.save('sub-'+str(paziente)+'_task-rsvp_epochs.fif', overwrite=True)
    
    return data, df_events, BU_categories, categories, new_events, obj_num, prova


home_path = os.path.abspath(os.getcwd())

#for ii in range(41, 51):
data, df_events, categories, tabellone, new_events, obj_num, prova = Start(4, home_path)
        
    
    
    
    
    
    
    
    
    
    
    
    