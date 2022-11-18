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

def averaging(epo):
    channels=dict(Occipital=[44, 45, 46, 47, 48, 15, 17, 16])
    evo=epo.average()
    temp=mne.channels.combine_channels(evo, channels,'median')
    res=temp.to_data_frame()
    datagrid=skfda.FDataGrid(res['Occipital'], res['time'])
    datagrid.plot()
    return datagrid
    
home_path = os.path.abspath(os.getcwd())
directory = home_path + '\preprocessed'
animal_ext=['animal','bird','insect']
food_ext=['food','dessert','drink','fruit','vegetable']
tool_ext=['tool','electronic device','kitchen appliance','kitchen tool','medical equipment','office supply','sports equipment','weapon']
for file in os.listdir(directory):
    filename = os.path.join(directory, file)
    epo = mne.read_epochs(filename)
    mne.epochs.combine_event_ids(epo,animal_ext,{'animal_ext':30},copy=False)
    mne.epochs.combine_event_ids(epo,food_ext,{'food_ext':31},copy=False)
    mne.epochs.combine_event_ids(epo,tool_ext,{'tool_ext':32},copy=False)
    
    animal = averaging(epo['animal_ext'])
    body = averaging(epo['body_part'])
    food = averaging(epo['food_ext'])
    vehicle = averaging(epo['vehicle'])
    tool = averaging(epo['tool_ext'])
    
    break
    


    
    