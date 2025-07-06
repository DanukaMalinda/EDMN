
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
    url = "../data/theorem/all-data-raw.csv"

    colnames = ["att" + str(i + 1) for i in range(58)]

    dta = pd.read_csv(url,
                      header=None,
                      names=colnames,
                      skiprows=1,
                      skipinitialspace=True)

    dta[dta == -100] = 1000
    dta["min_time"] = dta.iloc[:, -5:].min(axis=1)
    dta["res"] = (dta["min_time"] > 0).astype(int)

    dta = dta.drop(["att54", "att55", "att56", "att57", "att58", "min_time", "att5", "att35"], axis=1)

    if binned:
        for col in list(dta)[:-1]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
