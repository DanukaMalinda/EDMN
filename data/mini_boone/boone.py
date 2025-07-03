from data.mini_boone import prep
import numpy as np

def load_data():
    dataset = prep.prep_data()
    print(dataset.head())

    # dataset
    X_n = dataset.drop(columns=['signal']).values.astype(np.float32)
    y_n = dataset['signal'].values.astype(np.int64)

    print('y_n unique: ', np.unique(y_n))

    # Step 1: Compute mean and std from training data
    mean = X_n.mean(axis=0)
    std = X_n.std(axis=0) + 1e-8  # add epsilon to avoid divide-by-zero

    print('mean and std during data prep ',mean, std)

    # Step 2: Standardize
    X_train_std = (X_n - mean) / std

    X_train_std = np.nan_to_num(X_train_std, nan=0.0)

    print('mean and std: ',X_train_std.mean(), X_train_std.std())

    return X_train_std, y_n