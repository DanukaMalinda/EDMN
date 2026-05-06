# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:40:14 2022

@author: Zahra
"""

import pandas as pd
import os

def prep_data(binned=False):
    directory = os.getcwd()
    # url = directory + "/DryBeans.csv"
    url = "../data/dry_beans/DryBeans.csv"

    dta = pd.read_csv(url,
                      index_col=False,
                      skipinitialspace=True)

    
    dta.Class = dta.Class.replace({'BARBUNYA': 0, 'BOMBAY': 1, 'CALI': 2, 
                               'DERMASON': 3, 'HOROZ': 4, 'SEKER': 5, 'SIRA': 6})
    
    return dta