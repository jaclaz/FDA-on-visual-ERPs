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
    
    return final_raw

def column(matrix, i):
    return [row[i] for row in matrix]

def Start(paziente, home_path):
    
    categories = pd.read_csv(r"C:\Users\Asus\Downloads\things_concepts.tsv", sep='\t')
    
    if paziente<10:
        filename = home_path+"\sub-0"+str(paziente)+"\eeg\sub-0"+str(paziente)+"_task-rsvp_events.csv"
    if paziente>=10:
        filename = home_path+"\sub-"+str(paziente)+"\eeg\sub-"+str(paziente)+"_task-rsvp_events.csv"        
    df_events = pd.read_csv(filename)
    
    
    if paziente<10:
        filename = home_path+"\sub-0"+str(paziente)+"\eeg\sub-0"+str(paziente)+"_task-rsvp_eeg.vhdr"
    if paziente>=10:
        filename = home_path+"\sub-"+str(paziente)+"\eeg\sub-"+str(paziente)+"_task-rsvp_eeg.vhdr"
    
    
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
    num=obj_num.to_numpy()
    
    #imposto come codice evento i numeri associati ai concepts di THINGS
    
    for ii in np.arange(len(new_events[:])):
        new_events[ii][2]=num[ii]
        
    
    #nuovo dizionario per gli events con i concepts associati alle macro-categorie
    cat_dataframe = pd.read_csv(home_path+'\cat_dataframe.csv',index_col=0)
    new_dict=cat_dataframe.T.to_dict(orient='list')
    clean_dict={}
    
    for key_name in new_dict:
        items_cleaned = [item for item in new_dict[key_name] if not np.isnan(item)]
        clean_dict[key_name] = items_cleaned
        clean_dict[key_name] = [int(val) for val in clean_dict[key_name]]
    
    j=0
    for ii in column(new_events,2):
        k=0
        for key_name in clean_dict:
            if ii in clean_dict[key_name]:
                new_events[j][2]=k
            k+=1
        j+=1   
        
    print(new_events)    
    k=0
    for key_name in clean_dict:
        clean_dict[key_name]=k
        k+=1
    
    clean_dict['trial']=-1
        
    data.event_id=clean_dict

    #elimino le epochs con ptp amplitude > 150 microvolt
    reject_criteria = dict(eeg=150e-6)
    data.drop_bad(reject=reject_criteria)
    data.plot_drop_log()
    
    pazienti_no=[1, 6, 18, 23]
    if data.drop_log_stats()<80 and paziente not in pazienti_no:
        data.save('preprocessed\sub-'+str(paziente)+'_task-rsvp-epo.fif', overwrite=True)
    
    return data, cat_dataframe, clean_dict, new_dict, new_events, df_events,num


home_path = os.path.abspath(os.getcwd())

for ii in range(1, 51):
    data, prova, dictio,a,b,c,num = Start(ii, home_path)
        
    
    
    
    
    
    
    
    
    
    
    
    