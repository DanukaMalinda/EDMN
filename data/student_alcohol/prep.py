
import pandas as pd


def prep_data(binned=False):
    url1 = "../data/student_alcohol/student-mat.csv"
    url2 = "../data/student_alcohol/student-por.csv"

    dta1 = pd.read_csv(url1,
                       header=0,
                       skipinitialspace=True)

    dta2 = pd.read_csv(url2,
                       header=0,
                       skipinitialspace=True)

    dta = pd.concat([dta1, dta2], ignore_index=True)

    dta.sex = dta.sex.replace({"M": 0, "F": 1})
    dta = pd.get_dummies(dta)

    if binned:
        for col in ["age", "absences", "G1", "G2", "G3"]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
