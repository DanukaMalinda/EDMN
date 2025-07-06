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

def prep_data():
    url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00367/dota2Dataset.zip")

    my_zip_file = ZipFile(BytesIO(url.read()))
    train_file = my_zip_file.namelist()[0]
    test_file = my_zip_file.namelist()[1]

    hero_cols = ["H" + str(i + 1) for i in range(113)]
    colnames = ["Winner", "LocID", "GameMode", "GameType"] + hero_cols

    df_train = pd.read_csv(my_zip_file.open(train_file),
                           names=colnames,
                           skipinitialspace=True)

    df_test = pd.read_csv(my_zip_file.open(test_file),
                          names=colnames,
                          skipinitialspace=True)
    
    

    # dta = df_train.append(df_test, ignore_index=True)
    dta = pd.concat([df_train, df_test], ignore_index=True)

    dta = dta.drop(columns="LocID")

    dta = pd.get_dummies(dta, columns=["GameMode", "GameType"])

    dta.Winner = dta.Winner.replace({-1: 0,
                               1: 1})

    return dta

# dta.to_pickle("dta.pkl")
