# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:17:15 2025

@author: Danuka
"""

import pandas as pd
# from sklearn import preprocessing
# import os

def prep_data(binned=False):
    # directory = os.getcwd()
    url = "../data/microbes/microbes.csv"

    dta = pd.read_csv(url,
                      index_col=False,
                      skipinitialspace=True)
    
    dta.microorganisms = dta.microorganisms.replace({"Spirogyra": 0, "Ulothrix": 1, "Pithophora": 2, "Yeast": 3, "Raizopus": 4, "Penicillum": 5, "Aspergillus sp": 6, "Protozoa": 7, "Diatom": 8, "Volvox": 9,})
    # values = ['Spirogyra','Ulothrix','Volvox']
    # dta = dta[dta.target.isin(values) == False]
    # dta = dta.reset_index(drop=True)
    
    #transform categorical columns to numeric
    # labelencoder = preprocessing.LabelEncoder()

    # objFeatures = dta.select_dtypes(include="object").columns

    # for feat in objFeatures:
    #     dta[feat] = labelencoder.fit_transform(dta[feat].astype(str))
        
        
    return dta