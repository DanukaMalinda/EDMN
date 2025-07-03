# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:04:35 2022

@author: Zahra
"""

import pandas as pd
# import os

def prep_data(binned=False):
    # directory = os.getcwd()
    url = "../data/pendigits/pendigits.csv"

    dta = pd.read_csv(url,
                      index_col=False,
                      skipinitialspace=True)

    
    return dta
