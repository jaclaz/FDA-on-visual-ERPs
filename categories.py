# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:20:25 2022

@author: Asus
"""

import pandas as pd
import scipy.io
import numpy as np

def categorization(words,categories_matrix):

    categories = pd.read_csv(r"C:\Users\Asus\Downloads\things_concepts.tsv", sep='\t')
    BU_categories = categories['All Bottom-up Categories']
    
    res=[]
    k=0
    for cat in BU_categories:
        for  word in words:
            if word in cat:
                res.append(k)
        k+=1
    categories_matrix.append(res)
    return res

def find(target, matrix):
    for ii in np.arange(len(matrix)):
        for idx in matrix[ii]:
            if idx == target:
                return True
    return False

def DuplicateRemoval(matrix):
    for ii in np.arange(len(matrix)):
        nodup = list(dict.fromkeys(matrix[ii]))
        matrix[ii]=nodup
        
def CreateMissingMatrix(BU_categories,categories_matrix):
    missing=[]
    res=[]
    for idx in np.arange(len(BU_categories)):
        if not find(idx,categories_matrix):
            missing.append(BU_categories[idx])
            res.append(idx)
    categories_matrix.append(res)
    return missing,res

categories_name=['animal','bird','body part','clothing','clothing accessory','container','dessert','drink',
'electronic device','food','fruit','furniture','home decor','insect','kitchen appliance','kitchen tool','medical equipment',
'musical instrument','office supply','part of car','plant','sports equipment','tool','toy','vegetable','vehicle','weapon','uncategorized']

categories_matrix=[]
animal_idx=categorization(['animal','sea creature','fish','dog','reptile','crustacean','amphibian','pet','rodent'],categories_matrix)
bird_idx=categorization(['bird'],categories_matrix)
body_part_idx=categorization(['body part'],categories_matrix)
clothing_idx=categorization(['clothing','hat','shoe','footwear','underwear'],categories_matrix)
clothing_acc_idx=categorization(['clothing accessory','belt'],categories_matrix)
container_idx=categorization(['container','pet supply','smoking accessory'],categories_matrix)
dessert_idx=categorization(['dessert','pastry','cookie','candy'],categories_matrix)
drink_idx=categorization(['drink'],categories_matrix)
electronic_dev_idx=categorization(['electronic device','receiver'],categories_matrix)
food_idx=categorization(['food','herb','meal','bread','snack','meat','condiment'],categories_matrix)
fruit_idx=categorization(['fruit','nut'],categories_matrix)
furniture_idx=categorization(['furniture'],categories_matrix)
home_decor_idx=categorization(['home decor'],categories_matrix)
insect_idx=categorization(['insect','larva'],categories_matrix)
kitchen_app_idx=categorization(['kitchen appliance'],categories_matrix)
kitchen_tool_idx=categorization(['kitchen tool'],categories_matrix)
medical_eq_idx=categorization(['medical equipment'],categories_matrix)
musical_inst_idx=categorization(['musical instrument','music equipment'],categories_matrix)
office_supply_idx=categorization(['office supply'],categories_matrix)
part_of_car_idx=categorization(['part of car'],categories_matrix)
plant_idx=categorization(['plant','flower','weed','fungus','tree','seed'],categories_matrix)
sports_eq_idx=categorization(['sports equipment','hiking gear','gymnastics equipment','exercise equipment','fishing equipment','swimming equipment','baseball accessory'],categories_matrix)
tool_idx=categorization(['tool','mathematical device','photography equipment','scientific equipment','clock','rope'],categories_matrix)
toy_idx=categorization(['toy','game'],categories_matrix)
vegetable_idx=categorization(['vegetable'],categories_matrix)
vehicle_idx=categorization(['vehicle','watercraft','military vessel','boat','aircraft'],categories_matrix)
weapon_idx=categorization(['weapon','ammunition','gun accessory'],categories_matrix)

categories = pd.read_csv(r"C:\Users\Asus\Downloads\things_concepts.tsv", sep='\t')
BU_categories = categories['All Bottom-up Categories']

#gli indici delle immagini non ancora categorizzate
missing, missing_idx=CreateMissingMatrix(BU_categories,categories_matrix)

#rimuovo i duplicati e salvo come dataframe per usarlo con le epochs
DuplicateRemoval(categories_matrix)  
categories_df=pd.DataFrame(categories_matrix,categories_name)
categories_df.to_csv('cat_dataframe.csv')








