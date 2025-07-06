# -*- coding: utf-8 -*-
"""
data preparation.
Reference:
    - https://github.com/z-donyavi/MC-SQ/blob/main/helpers.py
    Z. Donyavi, A. B. S. Serapi˜ao, and G. Batista, “Mc-sq and mc-mq: Ensembles for multiclass quantification,
    ” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 8, pp. 4007–4019, 2024.

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
