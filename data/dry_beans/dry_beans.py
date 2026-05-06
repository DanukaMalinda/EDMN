from data.dry_beans import prep
import numpy as np

def load_data():
    dataset = prep.prep_data(True)
    print(dataset.head())

    # dataset
    X_n = dataset.iloc[:, :-1].values.astype(np.float32)
    y_n = dataset.iloc[:, -1].values.astype(np.int64)

    return X_n, y_n