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

def prep_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

    colnames = ["buying",
                "maint",
                "doors",
                "persons",
                "lug_boot",
                "safety",
                "acc_class"]

    dta = pd.read_csv(url,
                      names=colnames,
                      index_col=False,
                      skipinitialspace=True)

    dta.acc_class = dta.acc_class.replace({"unacc": 0,
                                           "acc": 1,
                                           "good": 1,
                                           "vgood": 1})

    dta = pd.get_dummies(dta)

    return dta

# dta.to_pickle("dta.pkl")