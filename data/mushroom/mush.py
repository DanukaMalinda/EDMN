from data.mushroom import prep
import numpy as np

def load_data():
    dataset = prep.prep_data()
    print(dataset.head())

    # dataset
    X_n = dataset.drop(columns=['result']).values.astype(np.float32)
    y_n = dataset['result'].values.astype(np.int64)

    # Step 1: Compute mean and std from training data
    mean = X_n.mean(axis=0)
    std = X_n.std(axis=0) + 1e-8  # add epsilon to avoid divide-by-zero

    # Step 2: Standardize
    X_train_std = (X_n - mean) / std
    X_train_std = np.nan_to_num(X_train_std, nan=0.0)

    print('mean and std: ',X_train_std.mean(), X_train_std.std())

    return X_train_std, y_n