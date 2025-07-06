# -*- coding: utf-8 -*-

"""
data preparation.
Reference:
    - https://github.com/z-donyavi/MC-SQ/blob/main/helpers.py
    Z. Donyavi, A. B. S. Serapi˜ao, and G. Batista, “Mc-sq and mc-mq: Ensembles for multiclass quantification,
    ” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 8, pp. 4007–4019, 2024.


"""

import pandas as pd
# import numpy as np
# import random
import os

def prep_data(binned=False):

    url = "../data/rice/rice.csv"

    dta = pd.read_csv(url,
                      index_col=False,
                      skipinitialspace=True)
    
    dta.CLASS = dta.CLASS.replace({'Arborio': 0, 'Basmati': 1, 'Ipsala': 2, 
                               'Jasmine': 3, 'Karacadag': 4})
    
    
    
    # random.seed(10)
    
    # size = 5000


    # # using groupby and some fancy logic
    
    # stratified = dta.groupby('Class', group_keys=False)\
    #                         .apply(lambda x: \
    #                          x.sample(int(np.rint(size*len(x)/len(dta)))))\
    #                         .sample(frac=1).reset_index(drop=True)

    return dta