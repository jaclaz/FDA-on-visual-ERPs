# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:47:48 2022

@author: Asus
"""

import numpy as np
import os
import pandas as pd
import mne

home_path = os.path.abspath(os.getcwd())
directory = home_path + '\preprocessed'

for file in os.listdir(directory):
    filename = os.path.join(directory, file)
    epo = mne.read_epochs(filename)
    break