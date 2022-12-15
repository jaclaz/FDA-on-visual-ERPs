# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:50:33 2022

@author: Asus
"""

import numpy as np
import os
import pandas as pd
import mne
import skfda
import matplotlib
import matplotlib.pyplot as plt

plt.ioff()
home_path = os.path.abspath(os.getcwd())
directory = 'E:\\Datasets\\NL3\\autoreject'
animal_ext=['animal','bird','insect']
food_ext=['food','dessert','drink','fruit','vegetable']
tool_ext=['tool','electronic device','kitchen appliance','kitchen tool',
          'medical equipment','office supply','sports equipment','weapon']

channels=['Fp1','Fz','F3','F7','FT9','FC5','FC1','C3', 'Oz']
#,'T7','TP9','CP5','CP1',
#             'Pz','P3','P7','O1','Oz','O2','P4','P8','TP10','CP6','CP2','C4','T8',
#             'FT10','FC6','FC2','F4','F8','Fp2','AF7','AF3','AFz','F1','F5','FT7',
#             'FC3','C1','C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4','PO8',
#            'P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6','AF8','AF4','F2','FCz']

directory_csv=home_path + '\dataframes'
    
def basisrep(cat, ch):
    df=pd.read_csv(home_path+'\\dataframes\\'+cat+'\\'+cat+'_'+ch+'.csv')
    data=df.iloc[:,2:]
    datagrid=skfda.FDataGrid(data_matrix=data.T)
    basis = skfda.representation.basis.BSpline(n_basis=18)
    smoother=skfda.preprocessing.smoothing.BasisSmoother(basis, smoothing_parameter=1e-5)
    cat_basis = smoother.fit_transform(datagrid)
    #cat_basis = datagrid.to_basis(basis)
    return cat_basis

def erp(epo, canale):
    evo=epo.average()
    return evo

for ch in ['Oz']:
    # fig1, axes1 = plt.subplots(nrows=2, ncols=1, num=1)
    # fig2, axes2 = plt.subplots(nrows=2, ncols=1, num=2)
    # fig3, axes3 = plt.subplots(nrows=2, ncols=1, num=3)
    # fig4, axes4 = plt.subplots(nrows=2, ncols=1, num=4)
    # fig5, axes5 = plt.subplots(nrows=2, ncols=1, num=5)
    # plt.figure(1)
    # axes1[1].set_title('Smoothed ERP Channel ' + ch + ' animal')
    # plt.figure(2)
    # axes2[1].set_title('Smoothed ERP Channel ' + ch + ' body part')
    # plt.figure(3)
    # axes3[1].set_title('Smoothed ERP Channel ' + ch + ' food')
    # plt.figure(4)
    # axes4[1].set_title('Smoothed ERP Channel ' + ch + ' vehicle')
    # plt.figure(5)
    # axes5[1].set_title('Smoothed ERP Channel ' + ch + ' tool')
    
    for file in os.listdir(directory):
        #if (file == 'sub-49_task-rsvp-epo.fif' and file != 'sub-50_task-rsvp-epo.fif'):
            filename = os.path.join(directory, file)
            epo = mne.read_epochs(filename)
            mne.epochs.combine_event_ids(epo,animal_ext,{'animal_ext':30},copy=False)
            mne.epochs.combine_event_ids(epo,food_ext,{'food_ext':31},copy=False)
            mne.epochs.combine_event_ids(epo,tool_ext,{'tool_ext':32},copy=False)
            
            animal=erp(epo['animal_ext'],ch)
            body=erp(epo['body part'],ch)
            food=erp(epo['food_ext'],ch)
            vehicle=erp(epo['vehicle'],ch)
            tool=erp(epo['tool_ext'],ch)
        
            plt.figure(1)
            animal.plot(picks=ch)
            # animal.plot(picks=ch, axes=axes1[0], show=False, selectable=False)
            # plt.figure(2)
            # body.plot(picks=ch, axes=axes2[0], show=False, selectable=False)
            # plt.figure(3)
            # food.plot(picks=ch, axes=axes3[0], show=False, selectable=False)
            # plt.figure(4)
            # vehicle.plot(picks=ch, axes=axes4[0], show=False, selectable=False)
            # plt.figure(5)
            # tool.plot(picks=ch, axes=axes5[0], show=False, selectable=False)
        
    # cat='animal'
    # cat_basis=basisrep(cat, ch)
    # plt.figure(1)
    # cat_basis.plot(axes=axes1[1])
    # fig1.tight_layout()
    # plt.savefig(home_path + '\\Plots\\Prova\\'+cat+'\\'+ch+'.png',dpi=160)
    
    # cat='body'
    # cat_basis=basisrep(cat, ch)
    # plt.figure(2)
    # cat_basis.plot(axes=axes2[1])
    # fig2.tight_layout()
    # plt.savefig(home_path + '\\Plots\\Prova\\'+cat+'\\'+ch+'.png',dpi=160)

    # cat='food'
    # cat_basis=basisrep(cat, ch)
    # plt.figure(3)
    # cat_basis.plot(axes=axes3[1])
    # fig3.tight_layout()
    # plt.savefig(home_path + '\\Plots\\Prova\\'+cat+'\\'+ch+'.png',dpi=160)
    
    # cat='vehicle'
    # cat_basis=basisrep(cat, ch)
    # plt.figure(4)
    # cat_basis.plot(axes=axes4[1])
    # fig4.tight_layout()
    # plt.savefig(home_path + '\\Plots\\Prova\\'+cat+'\\'+ch+'.png',dpi=160)
    
    # cat='tool'
    # cat_basis=basisrep(cat, ch)
    # plt.figure(5)
    # cat_basis.plot(axes=axes5[1])
    # fig5.tight_layout()
    # plt.savefig(home_path + '\\Plots\\Prova\\'+cat+'\\'+ch+'.png',dpi=160)
    
    # plt.show()
    # plt.close()

    