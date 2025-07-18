
from io import BytesIO
from zipfile import ZipFile
import urllib.request
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
    url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00304/BlogFeedback.zip")

    my_zip_file = ZipFile(BytesIO(url.read()))
    f = my_zip_file.namelist()[-1]

    dta = pd.read_csv(my_zip_file.open(f),
                      header=None,
                      names=["att" + str(i) for i in range(281)],
                      skipinitialspace=True)

    constcols = dta.columns[dta.nunique() == 1]
    dta = dta.drop(constcols, axis=1)

    bins = [-1, 0, 1, 10, 2000]
    labels = [0, 1, 2, 3]
    dta['att280'] = pd.cut(dta['att280'], bins=bins, labels=labels)
    dta['att280'] = dta['att280'].astype("int64")

    if binned:
        for col in list(dta)[:59]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        for col in list(dta)[272:-1]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
