
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
    url = "../data/student_performance/StudentsPerformance.csv"

    dta = pd.read_csv(url,
                      header=0,
                      skipinitialspace=True)

    dta = pd.get_dummies(dta)

    bins = [-1, 66, 100]
    labels = [0, 1]
    dta['math score'] = pd.cut(dta['math score'], bins=bins, labels=labels)
    dta['math score'] = dta['math score'].astype("int64")

    if binned:
        for col in list(dta)[1:3]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
