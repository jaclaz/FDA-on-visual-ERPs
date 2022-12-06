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
from skfda.exploratory.visualization import Boxplot
from skfda.inference.anova import oneway_anova

def erp(epo, canale, df,idx):
    evo=epo.average()
    temp=evo.pick_channels([canale])
    res=temp.to_data_frame()
    #evo.plot(picks=canale)
    
    if df.empty:
        df['time']=res['time']
    
    df['sub '+str(idx)] = res[canale]
     
home_path = os.path.abspath(os.getcwd())
directory = home_path + '\\autoreject'
animal_ext=['animal','bird','insect']
food_ext=['food','dessert','drink','fruit','vegetable']
tool_ext=['tool','electronic device','kitchen appliance','kitchen tool',
          'medical equipment','office supply','sports equipment','weapon']

channels=['Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','TP9','CP5','CP1',
          'Pz','P3','P7','O1','Oz','O2','P4','P8','TP10','CP6','CP2','C4','T8',
          'FT10','FC6','FC2','F4','F8','Fp2','AF7','AF3','AFz','F1','F5','FT7',
          'FC3','C1','C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4','PO8',
          'P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6','AF8','AF4',
          'F2','FCz']

#channels=['Oz']
directory_csv=home_path + '\dataframes'
    
# for ch in channels:
#     idx=0
#     animal=pd.DataFrame()
#     body=pd.DataFrame()
#     food=pd.DataFrame()
#     vehicle=pd.DataFrame()
#     tool=pd.DataFrame()
    
#     for file in os.listdir(directory):
#         if ((file == 'sub-49_task-rsvp-epo.fif' or file == 'sub-50_task-rsvp-epo.fif') and ch == 'Oz'):
#             break
#         filename = os.path.join(directory, file)
#         epo = mne.read_epochs(filename)
#         mne.epochs.combine_event_ids(epo,animal_ext,{'animal_ext':30},copy=False)
#         mne.epochs.combine_event_ids(epo,food_ext,{'food_ext':31},copy=False)
#         mne.epochs.combine_event_ids(epo,tool_ext,{'tool_ext':32},copy=False)
    
#         erp(epo['animal_ext'],ch,animal,idx)
#         erp(epo['body part'],ch,body,idx)
#         erp(epo['food_ext'],ch,food,idx)
#         erp(epo['vehicle'],ch,vehicle,idx)
#         erp(epo['tool_ext'],ch,tool,idx)
        
#         idx+=1

#     animal.to_csv('dataframes\\animal\\animal_'+ch+'.csv')
#     body.to_csv('dataframes\\body\\body_'+ch+'.csv')
#     food.to_csv('dataframes\\food\\food_'+ch+'.csv')
#     vehicle.to_csv('dataframes\\vehicle\\vehicle_'+ch+'.csv')
#     tool.to_csv('dataframes\\tool\\tool_'+ch+'.csv')

for cat in ['animal','body','vehicle','tool','food']:
    for ch in channels:
        df=pd.read_csv(home_path+'\\dataframes\\'+cat+'\\'+cat+'_'+ch+'.csv')
        data=df.iloc[:,2:]
        datagrid=skfda.FDataGrid(data_matrix=data.T)
        basis = skfda.representation.basis.BSpline(n_basis=10)
        cat_basis = datagrid.to_basis(basis)
        fig=cat_basis.plot()
        plt.title('Channel ' + ch + ' ' + cat)
        plt.savefig(home_path + '\\Plots\\basis_rep\\'+cat+'\\'+ch,dpi=160)
        plt.close()
    
        fdBoxplot = Boxplot(cat_basis.to_grid())
        fdBoxplot.show_full_outliers = True

        fig=fdBoxplot.plot()
        plt.title('Channel ' + ch + ' ' + cat)
        plt.savefig(home_path + '\\Plots\\boxplots\\'+cat+'\\'+ch,dpi=160)
        plt.close()
        
def Basis_rep(cat,ch):
    df=pd.read_csv(home_path+'\\dataframes\\'+cat+'\\'+cat+'_'+ch+'.csv')
    data=df.iloc[:,2:]
    datagrid=skfda.FDataGrid(data_matrix=data.T)
    basis = skfda.representation.basis.BSpline(n_basis=10)
    cat_basis = datagrid.to_basis(basis)
    
    # fdBoxplot = Boxplot(cat_basis.to_grid())
    # non_outliers = [i for (i, b) in zip(np.arange(len(fdBoxplot.outliers)), fdBoxplot.outliers) if not b]
    # cat_basis=cat_basis[non_outliers]
    return cat_basis

p_val=pd.DataFrame(index=channels,columns=['p_val'])

for ch in channels:
    animal=Basis_rep('animal', ch)
    body=Basis_rep('body', ch)
    vehicle=Basis_rep('vehicle', ch)
    tool=Basis_rep('tool', ch)
    food=Basis_rep('food', ch)
    
    v_n, p = oneway_anova(animal, body, vehicle, tool, food)
    
    p_val.loc[ch]=p
    
p_val.to_csv('p_values.csv')

def reversed_enumerate(l):
    return zip(range(len(l)-1, -1, -1), reversed(l))

def Hochberg(p_val_passed, alpha = 0.05, debug = False):

    p_val = p_val_passed.copy() # copy of the list to not modify it

    m = len(p_val)
    
    order = p_val.argsort()
    p_val = p_val[order]

    if debug:
        print('m = ',m,'\norder: ',order,'\npval sorted: ',p_val,'\n\nFor loop: \n')

    for i,p in reversed_enumerate(p_val):
        treshold = alpha/(m - i)    # alpha \ (m - (i + 1) + 1)
        if debug:
            print('i ordered = ',order[i],'\tpval = ',p,'\t treshold = ',treshold)
        if(p <= treshold):
            break

    if i == 0:
        idx_reject = []
    else:
        idx_reject = order[:(i+1)]
    
    if debug:
        print('\nindexes rejected: ', idx_reject)

    return idx_reject

indxs_Hoch = Hochberg(p_val.to_numpy().flatten(), debug = True)
indxs_Hoch
    
ch_rej=list(channels[i] for i in np.sort(indxs_Hoch))
ch_rej
    