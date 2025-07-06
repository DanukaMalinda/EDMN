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
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00262/turkiye-student-evaluation_generic.csv"

    dta = pd.read_csv(url,
                      index_col=False,
                      sep=',',
                      skipinitialspace=True)

    dta = dta.drop(columns=['class'])

    dta.instr = dta.instr.replace({1: 0,
                                   2: 0,
                                   3: 1})

    # dta.to_pickle("dta.pkl")

    return dta
