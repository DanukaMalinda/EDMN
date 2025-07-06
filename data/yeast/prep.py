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
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"

    colnames = ['name',
                'mcg',
                'gvh',
                'alm',
                'mit',
                'erl',
                'pox',
                'vac',
                'nuc',
                'class']

    dta = pd.read_csv(url,
                      names=colnames,
                      index_col=False,
                      sep=' ',
                      skipinitialspace=True)

    dta = dta.drop(columns=['name'])
    dta['class'] = dta['class'].replace({'CYT': 0,
                                         'NUC': 1,
                                         'MIT': 2,
                                         'ME3': 3,
                                         'ME2': 4,
                                         'ME1': 4,
                                         'EXC': 4,
                                         'VAC': 4,
                                         'POX': 4,
                                         'ERL': 4})
    dta = dta.loc[dta["class"] < 4]

    # dta.to_pickle("dta.pkl")

    if binned:
        bins = [0, 0.4, 0.5, 0.6, 1]
        labels = [1, 2, 3, 4]
        dta['mcg'] = pd.cut(dta['mcg'], bins=bins, labels=labels)
        dta['mcg'] = dta['mcg'].astype("int64")

        bins = [0, 0.4, 0.5, 0.6, 1]
        labels = [1, 2, 3, 4]
        dta['gvh'] = pd.cut(dta['gvh'], bins=bins, labels=labels)
        dta['gvh'] = dta['gvh'].astype("int64")

        bins = [0, 0.4, 0.5, 0.6, 1]
        labels = [1, 2, 3, 4]
        dta['alm'] = pd.cut(dta['alm'], bins=bins, labels=labels)
        dta['alm'] = dta['alm'].astype("int64")

        bins = [-0.1, 0.1, 0.2, 0.3, 1]
        labels = [1, 2, 3, 4]
        dta['mit'] = pd.cut(dta['mit'], bins=bins, labels=labels)
        dta['mit'] = dta['mit'].astype("int64")

        bins = [-0.1, 0.4, 0.5, 0.6, 1]
        labels = [1, 2, 3, 4]
        dta['pox'] = pd.cut(dta['pox'], bins=bins, labels=labels)
        dta['pox'] = dta['pox'].astype("int64")

        bins = [-0.1, 0.25, 0.35, 1]
        labels = [1, 2, 3]
        dta['vac'] = pd.cut(dta['vac'], bins=bins, labels=labels)
        dta['vac'] = dta['vac'].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta
