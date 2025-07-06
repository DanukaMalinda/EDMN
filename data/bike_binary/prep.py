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
    url = urllib.request.urlopen(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip")

    my_zip_file = ZipFile(BytesIO(url.read()))
    f = my_zip_file.namelist()[2]

    dta = pd.read_csv(my_zip_file.open(f),
                      header=0,
                      skipinitialspace=True)

    dta = dta.drop(["instant", "casual", "registered", "dteday"], axis=1)

    dta = pd.get_dummies(dta, columns=["season", "yr", "mnth", "hr", "weekday", "weathersit"])

    bins = [0, 100, 1000]
    labels = [0, 1]
    dta['cnt'] = pd.cut(dta['cnt'], bins=bins, labels=labels)
    dta['cnt'] = dta['cnt'].astype("int64")

    if binned:
        for col in list(dta)[2:6]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

            # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
