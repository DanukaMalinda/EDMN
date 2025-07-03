# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:54:16 2022

@author: Zahra
"""


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def prep_data(binned=False):
    
    url = "../data/gesture/gesture32.csv"

    dta = pd.read_csv(url,
                      index_col=False,
                      skipinitialspace=True)
    #transform categorical columns to numeric
    dta.Phase = dta.Phase.replace({"D": 0, "P": 1, "S": 2, "H": 3, "R": 4})

    # Separate label and features
    labels = dta['Phase']
    features = dta.drop('Phase', axis=1)

    # Normalize features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # Combine back label and normalized features
    normalized_df = pd.DataFrame(normalized_features, columns=features.columns)
    normalized_df.insert(0, 'Phase', labels)
    
    return dta

