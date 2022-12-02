# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:47:48 2022

@author: Asus
"""

import numpy as np
import os
import pandas as pd
import mne
import skfda
import matplotlib.pyplot as plt

def erp(epo, canale, df,idx):
    evo=epo.average()
    temp=evo.pick_channels([canale])
    res=temp.to_data_frame()
    
    if df.empty:
        df['time']=res['time']
    
    df['sub '+str(idx)] = res[canale]
     
home_path = os.path.abspath(os.getcwd())
directory = home_path + '\preprocessed'
animal_ext=['animal','bird','insect']
food_ext=['food','dessert','drink','fruit','vegetable']
tool_ext=['tool','electronic device','kitchen appliance','kitchen tool','medical equipment','office supply','sports equipment','weapon']

#channels=['Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','TP9','CP5','CP1','Pz','P3','P7']
#, 
channels=['O1','Oz','O2','P4','P8','TP10','CP6','CP2','C4','T8','FT10','FC6','FC2','F4','F8','Fp2']
#,
# 'AF7','AF3','AFz','F1','F5','FT7','FC3','C1','C5','TP7','CP3','P1','P5','PO7','PO3','POz',
# 'PO4','PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6','AF8','AF4','F2','FCz']

directory_csv=home_path + '\dataframes'
    
for ch in channels:
    idx=0
    animal=pd.DataFrame()
    body=pd.DataFrame()
    food=pd.DataFrame()
    vehicle=pd.DataFrame()
    tool=pd.DataFrame()
    
    for file in os.listdir(directory):

        filename = os.path.join(directory, file)
        epo = mne.read_epochs(filename)
        mne.epochs.combine_event_ids(epo,animal_ext,{'animal_ext':30},copy=False)
        mne.epochs.combine_event_ids(epo,food_ext,{'food_ext':31},copy=False)
        mne.epochs.combine_event_ids(epo,tool_ext,{'tool_ext':32},copy=False)
    
        erp(epo['animal_ext'],ch,animal,idx)
        erp(epo['body part'],ch,body,idx)
        erp(epo['food_ext'],ch,body,idx)
        erp(epo['vehicle'],ch,vehicle,idx)
        erp(epo['tool_ext'],ch,tool,idx)
        
        idx+=1
        
    animal.to_csv('dataframes\\animal\\animal_'+ch+'.csv')
    body.to_csv('dataframes\\body\\body_'+ch+'.csv')
    food.to_csv('dataframes\\food\\food_'+ch+'.csv')
    vehicle.to_csv('dataframes\\vehicle\\vehicle_'+ch+'.csv')
    tool.to_csv('dataframes\\tool\\tool_'+ch+'.csv')

for cat in ['animal','body','vehicle','tool']:
    df=pd.read_csv(home_path+'\\dataframes\\'+cat+'\\'+cat+'_Oz.csv')
    data=df.iloc[:,1:]
    datagrid=skfda.FDataGrid(data_matrix=data)
    #fig=datagrid.plot()
    basis = skfda.representation.basis.BSpline(n_basis=5)
    cat_basis = datagrid.to_basis(basis)
    fig=cat_basis.plot()
    plt.title('Channel Oz '+ cat)
    plt.show()
    


    
    