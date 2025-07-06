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
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

    colnames = ['result',
                'ap-shape',
                'ap-surface',
                'ap-color',
                'ruises?',
                'dor',
                'ill-attachment',
                'ill-spacing',
                'ill-size',
                'ill-color',
                'stalk-shape',
                'stalk-root',
                'stalk-surface-above-ring',
                'stalk-surface-below-ring',
                'stalk-color-above-ring',
                'stalk-color-below-ring',
                'veil-type',
                'veil-color',
                'ring-number',
                'ring-type',
                'spore-print-color',
                'population',
                'habitat']

    dta = pd.read_csv(url,
                      names=colnames,
                      index_col=False,
                      skipinitialspace=True)

    dta.result = dta.result.replace({"p": 0, "e": 1})
    dta['ring-number'] = dta['ring-number'].replace({"n": 0, "o": 1, "t": 2})
    dta = dta.drop(columns=['stalk-root'])
    dta = pd.get_dummies(dta)

    return dta

# dta.to_pickle("dta.pkl")
