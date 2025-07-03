# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:14:56 2022

@author: Zahra
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# import os

def prep_data(binned=False):
    # directory = os.getcwd()
    url = "../data/human_activity_recognition/hars.csv"

    dta = pd.read_csv(url,
                      index_col=False,
                      skipinitialspace=True)
    #transform categorical columns to numeric
    dta.Activity = dta.Activity.replace({"STANDING": 0, "SITTING": 1, "LAYING": 2, "WALKING": 3, "WALKING_DOWNSTAIRS": 4, "WALKING_UPSTAIRS": 5})

    # Separate label and features
    labels = dta['Activity']
    features = dta.drop('Activity', axis=1)

    # Normalize features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # Combine back label and normalized features
    normalized_df = pd.DataFrame(normalized_features, columns=features.columns)
    normalized_df.insert(0, 'Activity', labels)

    # labelencoder = preprocessing.LabelEncoder()

    # objFeatures = dta.select_dtypes(include="object").columns

    # for feat in objFeatures:
    #     dta[feat] = labelencoder.fit_transform(dta[feat].astype(str))
    
    return normalized_df
