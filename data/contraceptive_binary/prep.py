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
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data"
    colnames = ["Age", "Education", "HusbandEducation", "NumberChildren", "Islamic", "Work", "HusbandJob",
                "LivingStandard", "MediaExposure", "Contraceptive"]

    dta = pd.read_csv(url,
                      names=colnames,
                      skipinitialspace=True)

    dta = pd.get_dummies(dta, columns=["HusbandJob"])

    dta.Contraceptive = dta.Contraceptive.replace({1: 0,
                                                   2: 1,
                                                   3: 1})

    if binned:
        bins = [15, 25, 30, 35, 40, 50]
        labels = [1, 2, 3, 4, 5]
        dta['Age'] = pd.cut(dta['Age'], bins=bins, labels=labels)
        dta['Age'] = dta['Age'].astype("int64")

        bins = [-1, 0, 1, 2, 3, 5, 20]
        labels = [0, 1, 2, 3, 4, 5]
        dta['NumberChildren'] = pd.cut(dta['NumberChildren'], bins=bins, labels=labels)
        dta['NumberChildren'] = dta['NumberChildren'].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
