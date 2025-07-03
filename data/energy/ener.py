from data.energy import prep
import numpy as np
from scipy.stats import pearsonr

def load_data():
    dataset = prep.prep_data()
    print(dataset.head())

    # dataset
    X_n = dataset.drop(columns=['Appliances']).values.astype(np.float32)
    y_n = dataset['Appliances'].values.astype(np.int64)

    y_n =   y_n-1

    # Ensure y_n is a 1D array
    y = y_n.flatten()

    # Compute correlation between each feature and target
    correlations = []
    for i in range(X_n.shape[1]):
        corr, _ = pearsonr(X_n[:, i], y)
        correlations.append(corr)

    correlations = np.array(correlations)

    # Keep features with absolute Pearson correlation > 0.2
    threshold = 0.15
    selected_indices = np.where(np.abs(correlations) > threshold)[0]
    X_selected = X_n[:, selected_indices]

    # Step 1: Compute mean and std from training data
    mean = X_selected.mean(axis=0)
    std = X_selected.std(axis=0) + 1e-8  # add epsilon to avoid divide-by-zero

    # Step 2: Standardize
    X_train_std = (X_selected - mean) / std
    X_train_std = np.nan_to_num(X_train_std, nan=0.0)

    print('mean and std: ',X_train_std.mean(), X_train_std.std())

    return X_train_std, y_n