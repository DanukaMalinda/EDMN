
import pandas as pd

"""
data preparation.
Reference:
    - https://github.com/z-donyavi/MC-SQ/blob/main/helpers.py
    Z. Donyavi, A. B. S. Serapi˜ao, and G. Batista, “Mc-sq and mc-mq: Ensembles for multiclass quantification,
    ” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 8, pp. 4007–4019, 2024.

    - https://github.com/tobiasschumacher/quantification_paper/blob/master/helpers.py
    T. Schumacher, M. Strohmaier, and F. Lemmerich, 
    “A comparative evaluation of quantification methods,” arXiv preprint arXiv:2103.03223, 2021.

"""

def prep_data(binned=False):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data"

    colnames = ["att" + str(i) for i in range(32)]

    dta = pd.read_csv(url,
                      header=None,
                      names=colnames,
                      skipinitialspace=True)

    dta = dta.drop(["att0"], axis=1)

    # dta.loc[dta['Class'] != "A", 'Class'] = "B"
    dta.att28 = dta.att28.replace({"CL0": 0,
                                   "CL1": 1,
                                   "CL2": 1,
                                   "CL3": 1,
                                   "CL4": 1,
                                   "CL5": 1,
                                   "CL6": 1})

    dta = pd.get_dummies(dta)

    if binned:
        for col in list(dta)[:12]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")