from data.turk_student_eval_binary import prep
import numpy as np
from scipy.stats import pearsonr

def load_data():
    dataset = prep.prep_data()
    print(dataset.head())
    # Assuming df is your DataFrame
    question_cols = [f"Q{i}" for i in range(1, 29)]

    # Add sum and average columns
    dataset["Q_sum"] = dataset[question_cols].sum(axis=1)
    dataset["Q_avg"] = dataset[question_cols].mean(axis=1)
    

    # Step 1: Multiply the two columns
    dataset['product'] = dataset['nb.repeat'] * dataset['Q_sum']

    # Step 2: Calculate the total sum of this new column
    # Step 2: Calculate quartiles
    q1 = dataset['product'].quantile(0.25)
    q2 = dataset['product'].quantile(0.50)
    q3 = dataset['product'].quantile(0.75)

    # Step 3: Create labels based on the total
    def label_value(x):
        if x < q1:
            return 0
        elif x < q2:
            return 0
        elif x < q3:
            return 1
        else:
            return 1

    # Step 4: Apply the function to create a new label column
    dataset['label'] = dataset['product'].apply(label_value)
    print(dataset.head())

    # dataset.to_csv('turk.csv', index=False)

    # dataset
    X_n = dataset.drop(columns=['difficulty','product','label' ]).values.astype(np.float32)
    y_n = dataset['label'].values.astype(np.int64)



    # Step 1: Compute mean and std from training data
    mean = X_n.mean(axis=0)
    std = X_n.std(axis=0) + 1e-8  # add epsilon to avoid divide-by-zero

    # Step 2: Standardize
    X_train_std = (X_n - mean) / std
    X_train_std = np.nan_to_num(X_train_std, nan=0.0)

    print('mean and std: ',X_train_std.mean(), X_train_std.std())

    return X_train_std, y_n