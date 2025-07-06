

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
    url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip")

    my_zip_file = ZipFile(BytesIO(url.read()))
    f = my_zip_file.namelist()[1]

    dta = pd.read_csv(my_zip_file.open(f),
                      header=0,
                      skipinitialspace=True)

    bins = [0, 30, 200]
    labels = [0, 1]
    dta['critical_temp'] = pd.cut(dta['critical_temp'], bins=bins, labels=labels)
    dta['critical_temp'] = dta['critical_temp'].astype("int64")

    if binned:
        for col in list(dta)[1:-1]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

    dta = pd.get_dummies(dta, columns=["number_of_elements"])

    # dta.to_pickle("dta.pkl")
    # dta.to_pickle("dta_binned.pkl")

    return dta
