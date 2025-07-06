
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
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"

    colnames = ["att" + str(i + 1) for i in range(9)]

    dta = pd.read_csv(url,
                      header=None,
                      names=colnames,
                      skipinitialspace=True)

    dta.att9 = dta.att9.replace({"not_recom": 0, "recommend": 1, "very_recom": 1, "priority": 1, "spec_prior": 2})
    dta = pd.get_dummies(dta)

    return dta

# dta.to_pickle("dta.pkl")
