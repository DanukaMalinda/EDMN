import math
import numpy as np

"""
Synthetic data draw function for generating training and test indices based on class distributions.
Reference:
    - https://github.com/z-donyavi/MC-SQ/blob/main/helpers.py
    Z. Donyavi, A. B. S. Serapi˜ao, and G. Batista, “Mc-sq and mc-mq: Ensembles for multiclass quantification,
    ” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 8, pp. 4007–4019, 2024.

    - https://github.com/tobiasschumacher/quantification_paper/blob/master/helpers.py
    T. Schumacher, M. Strohmaier, and F. Lemmerich, 
    “A comparative evaluation of quantification methods,” arXiv preprint arXiv:2103.03223, 2021.

"""

def get_draw_size(y_cts, dt, train_distr, test_distr, C=None):
    if len(train_distr) != len(test_distr):
        raise ValueError("training and test distributions are not the same length")

    if C is None:
        C = sum(y_cts)

    constraints = [C] + [y_cts[i] / (dt[0] * train_distr[i] + dt[1] * test_distr[i])
                         for i in range(len(y_cts))]

    return np.floor(min(constraints))

def synthetic_draw(n_y, n_classes, y_cts, y_idx, dt_distr, train_distr, test_distr, seed=4711):
    if len(train_distr) != len(test_distr):
        raise ValueError("training and test distributions are not the same length")

    if len(y_cts) != len(train_distr):
        raise ValueError("Length of training distribution does not match number of classes")

    if len(dt_distr) != 2:
        raise ValueError("Length of train/test-split has to equal 2")

    if not math.isclose(np.sum(dt_distr), 1.0):
        raise ValueError("Elements of train/test-split do not sum to 1")

    if not math.isclose(np.sum(train_distr), 1.0):
        raise ValueError("Elements of train distribution do not sum to 1")

    if not math.isclose(np.sum(test_distr), 1.0):
        raise ValueError("Elements of test distribution do not sum to 1")

    n = get_draw_size(y_cts, dt_distr, train_distr, test_distr, C=n_y)

    train_cts = (n * dt_distr[0] * train_distr).astype(int)
    if min(train_cts) == 0:
        raise ValueError("Under given input distributions a class would miss in training")

    test_cts = (n * dt_distr[1] * test_distr).astype(int)

    # fix seed for reproducibility
    np.random.seed(seed)

    train_list = [np.random.choice(y_idx[i], size=train_cts[i], replace=False) for i in range(n_classes)]
    y_idx = [np.setdiff1d(y_idx[i], train_list[i]) for i in range(n_classes)]
    test_list = [np.random.choice(y_idx[i], size=test_cts[i], replace=False) if np.size(y_idx[i]) > 0 else [] for i in
                 range(n_classes)]

    train_index = np.concatenate(train_list)
    test_index = np.concatenate(test_list).astype(int)

    n_train = train_index.shape[0]
    n_test = test_index.shape[0]
    M = n_train + n_test
    r_train = n_train * 1.0 / M
    r_test = n_test * 1.0 / M

    train_ratios = train_cts * 1.0 / n_train
    test_ratios = test_cts * 1.0 / n_test

    stats_vec = np.concatenate(
        [np.array([M, n_train, n_test, r_train, r_test]), train_cts, train_ratios, test_cts, test_ratios])

    return train_index, test_index, stats_vec   