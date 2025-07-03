# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:42:27 2022

@author: Zahra
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# import os

def prep_data(binned=False):
    # directory = os.getcwd()
    url = "../data/gasdrift/gasdrift.csv"

    dta = pd.read_csv(url,
                      index_col=False,
                      skipinitialspace=True)
    
    # Separate label and features
    labels = dta['label']
    features = dta.drop('label', axis=1)

    # Normalize features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # Combine back label and normalized features
    normalized_df = pd.DataFrame(normalized_features, columns=features.columns)
    normalized_df.insert(0, 'label', labels)

    
    return normalized_df

