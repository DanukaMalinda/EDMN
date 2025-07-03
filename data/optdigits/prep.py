# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:53:54 2022

@author: Zahra
"""

import pandas as pd
# import os

def prep_data(binned=False):
    # directory = os.getcwd()
    url = "../data/optdigits/optdigits.csv"
    
    
    dta = pd.read_csv(url,
                      index_col=False,
                      skipinitialspace=True)
    

    # dta = dta.drop(["A1"], axis=1)
    
    # dta.target = dta.target.replace({1: 0, 3: 2, 5: 4,
    #                                  7: 6, 9: 8})
    
    # dta.target = dta.target.replace({6: 1, 8: 3})
    return dta