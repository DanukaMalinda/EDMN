"""
Run KDEy-ML on our datasets using exactly the same train/test sampling
protocol as run_ablation.py:
  - synthetic_draw() to create prevalence-controlled train/test splits
  - Loop over dt_distr × train_distr × test_distr configs
  - Leftover samples (natural prevalence) used to calibrate KDE
  - MAE = mean |estimated_prevalence - actual_test_prevalence| over classes

Usage:
    python run_quapy_comparison.py --dataset beans
    python run_quapy_comparison.py --all
"""

import sys
import os
import math
import argparse
import numpy as np

# --- Path setup ---
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(SRC_DIR)
QUAPY_DIR = os.path.join(ROOT_DIR, 'QuaPy')

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, QUAPY_DIR)

from quapy.data import LabelledCollection
from quapy.method._kdey import KDEyML
from sklearn.linear_model import LogisticRegression

import data_loader


def make_kdey_ml(X_leftover=None, y_leftover=None):
    """Return a fresh KDEy-ML instance.

    Uses leftover samples (natural prevalence, disjoint from train+test) for
    KDE calibration. Falls back to val_split=0.4 when no leftover is available.
    """
    val = (X_leftover, y_leftover) if (X_leftover is not None and len(X_leftover) > 0) else 0.4
    return KDEyML(LogisticRegression(max_iter=3000, C=1.0), val_split=val)

# --------------------------------------------------------------------------
# Prevalence tables (identical to run_ablation.py)
# --------------------------------------------------------------------------
train_2_prevalences  = [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]]
train_3_prevalences  = [[0.2, 0.5, 0.3], [0.05, 0.8, 0.15], [0.35, 0.3, 0.35]]
train_4_prevalences  = [[0.5, 0.3, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1],
                        [0.25, 0.25, 0.25, 0.25]]
train_5_prevalences  = [[0.2, 0.15, 0.35, 0.1, 0.2], [0.35, 0.25, 0.15, 0.15, 0.1],
                        [0.2, 0.2, 0.2, 0.2, 0.2]]
train_6_prevalences  = [[0.1, 0.2, 0.1, 0.1, 0.25, 0.25],
                        [0.05, 0.1, 0.3, 0.4, 0.1, 0.05],
                        [0.17, 0.17, 0.16, 0.17, 0.17, 0.16]]
train_7_prevalences  = [[0.2, 0.3, 0.2, 0.15, 0.05, 0.05, 0.05],
                        [0.05, 0.1, 0.05, 0.05, 0.25, 0.3, 0.2],
                        [0.15, 0.14, 0.14, 0.15, 0.14, 0.14, 0.14]]
train_10_prevalences = [[0.05, 0.2, 0.05, 0.1, 0.05, 0.25, 0.05, 0.05, 0.1, 0.1],
                        [0.1,  0.1, 0.1,  0.1, 0.1,  0.1,  0.1,  0.1,  0.1, 0.1],
                        [0.3,  0.1, 0.1,  0.1, 0.1,  0.1,  0.05, 0.05, 0.05, 0.05]]

test_2_prevalences  = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5],
                       [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]]
test_3_prevalences  = [[0.1, 0.7, 0.2], [0.55, 0.1, 0.35], [0.35, 0.55, 0.1],
                       [0.4, 0.25, 0.35], [0.0, 0.05, 0.95]]
test_4_prevalences  = [[0.65, 0.25, 0.05, 0.05], [0.2, 0.25, 0.3, 0.25],
                       [0.45, 0.15, 0.2, 0.2], [0.2, 0.0, 0.0, 0.8],
                       [0.3, 0.25, 0.35, 0.1]]
test_5_prevalences  = [[0.15, 0.1, 0.65, 0.1, 0], [0.45, 0.1, 0.3, 0.05, 0.1],
                       [0.2, 0.25, 0.25, 0.1, 0.2], [0.35, 0.05, 0.05, 0.05, 0.5],
                       [0.05, 0.25, 0.15, 0.15, 0.4]]
test_6_prevalences  = [[0.15, 0.1, 0.55, 0.1, 0.0, 0.1],
                       [0.4, 0.1, 0.25, 0.05, 0.1, 0.1],
                       [0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
                       [0.35, 0.05, 0.05, 0.05, 0.05, 0.45],
                       [0.05, 0.25, 0.15, 0.15, 0.1, 0.3]]
test_7_prevalences  = [[0.1, 0.1, 0.1, 0.5, 0.1, 0.0, 0.1],
                       [0.4, 0.1, 0.2, 0.05, 0.1, 0.1, 0.05],
                       [0.15, 0.2, 0.15, 0.1, 0.2, 0.1, 0.1],
                       [0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.45],
                       [0.05, 0.25, 0.1, 0.15, 0.1, 0.3, 0.05]]
test_10_prevalences = [[0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0, 0.1, 0.05, 0.05],
                       [0.05, 0.05, 0.1, 0.05, 0.05, 0.3, 0.1, 0.1, 0.1, 0.1],
                       [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                       [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05],
                       [0.0, 0.0, 0.2, 0.2, 0.3, 0.1, 0.1, 0.0, 0.05, 0.05]]

train_prev_dictionary = {2: train_2_prevalences, 3: train_3_prevalences,
                         4: train_4_prevalences, 5: train_5_prevalences,
                         6: train_6_prevalences, 7: train_7_prevalences,
                         10: train_10_prevalences}
test_prev_dictionary  = {2: test_2_prevalences,  3: test_3_prevalences,
                         4: test_4_prevalences,  5: test_5_prevalences,
                         6: test_6_prevalences,  7: test_7_prevalences,
                         10: test_10_prevalences}

train_test_ratios = [np.array(d) for d in [[0.9, 0.1], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3]]]

# --------------------------------------------------------------------------
# Synthetic draw — identical to run_ablation.py
# --------------------------------------------------------------------------
def get_draw_size(y_cts, dt, train_distr, test_distr, C=None):
    constraints = [C] + [
        y_cts[i] / (dt[0] * train_distr[i] + dt[1] * test_distr[i])
        for i in range(len(train_distr))
        if (dt[0] * train_distr[i] + dt[1] * test_distr[i]) > 0
    ]
    return np.floor(min(constraints))


def synthetic_draw(
        n_y, n_classes, y_cts, y_idx, dt_distr, train_distr, test_distr, seed=4711):
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

    np.random.seed(seed)
    train_list = [np.random.choice(y_idx[i], size=train_cts[i], replace=False)
                  for i in range(n_classes)]
    y_idx_rem  = [np.setdiff1d(y_idx[i], train_list[i]) for i in range(n_classes)]
    test_list  = [np.random.choice(y_idx_rem[i], size=test_cts[i], replace=False)
                  for i in range(n_classes)]

    train_index = np.concatenate(train_list)
    test_index  = np.concatenate(test_list).astype(int)

    # Leftover: samples not drawn into train or test — natural dataset prevalence
    leftover_index = np.concatenate([
        np.setdiff1d(y_idx_rem[i], test_list[i]) for i in range(n_classes)
    ])

    return train_index, test_index, leftover_index

# --------------------------------------------------------------------------
# Quantify one config with KDEy-ML, return MAE
# --------------------------------------------------------------------------
def quantify_config(quantifier, X_train, y_train, X_test, y_test, n_classes):
    quantifier.fit(X_train, y_train)
    estim_prev = quantifier.quantify(X_test)

    # Use observed proportions — matches EDMN's AE() which uses
    # get_actual_count(y_test), not the configured test_distr
    # (test_cts are integer-rounded so observed != configured exactly)
    test_lc   = LabelledCollection(X_test, y_test, classes=list(range(n_classes)))
    actual_obs = test_lc.prevalence()
    mae = float(np.mean(np.abs(estim_prev - actual_obs)))
    return mae


# --------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------
def run_dataset(dataset):
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset}")
    print(f"{'='*60}")

    X_n, y_n = data_loader.load_data(dataset)
    X_n = np.array(X_n, dtype=float)
    y_n = np.array(y_n, dtype=int)

    # Remap labels to 0..L-1
    classes = np.unique(y_n)
    if not np.array_equal(classes, np.arange(len(classes))):
        mapping = {c: i for i, c in enumerate(classes)}
        y_n = np.array([mapping[yi] for yi in y_n])

    Y           = np.unique(y_n)
    num_classes = len(Y)
    y_idx       = [np.where(y_n == l)[0] for l in Y]
    y_cts       = np.array([len(y_idx[i]) for i in range(num_classes)])
    N           = len(X_n)

    print(f"  N={N}  features={X_n.shape[1]}  classes={num_classes}  "
          f"counts={dict(zip(Y.tolist(), y_cts.tolist()))}")

    if num_classes not in train_prev_dictionary:
        raise ValueError(f"No prevalence table for {num_classes} classes")

    train_ds = np.array(train_prev_dictionary[num_classes])
    test_ds  = np.array(test_prev_dictionary[num_classes])

    maes      = []
    config_id = 0
    skipped   = 0

    for dt_distr in train_test_ratios:
        for train_distr in train_ds:
            for test_distr in test_ds:
                config_id += 1
                try:
                    train_index, test_index, leftover_index = synthetic_draw(
                        N, num_classes, y_cts, y_idx,
                        dt_distr, train_distr, test_distr, seed=11)

                    X_train,    y_train    = X_n[train_index],    y_n[train_index]
                    X_test,     y_test     = X_n[test_index],     y_n[test_index]
                    X_leftover, y_leftover = X_n[leftover_index], y_n[leftover_index]

                    quantifier = make_kdey_ml(X_leftover, y_leftover)
                    mae = quantify_config(quantifier, X_train, y_train,
                                         X_test, y_test, num_classes)
                    maes.append(mae)

                except Exception:
                    skipped += 1
                    continue

    print(f"\n  Configs: {config_id} total, {skipped} skipped")
    if maes:
        print(f"  KDEy-ML  Mean MAE: {np.mean(maes):.5f}  ({len(maes)} configs)")
    else:
        print(f"  KDEy-ML  ERROR: no valid configs")

    return np.mean(maes) if maes else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--all',     action='store_true')
    args = parser.parse_args()

    if args.dataset is None and not args.all:
        parser.error("Provide --dataset <name> or --all")

    datasets = [
        'beans', 'conc',  'drugs', 'yeast', 'gest',  'hars',
        'micro', 'wine',  'turk',  'bike',  'blog',  'diam',
        'ener',  'fifa',  'gasd',  'news',  'nurse', 'optd',
        'pend',  'rice',  'cappl', 'vgame',
    ] if args.all else [args.dataset]

    all_results = {}
    for ds in datasets:
        try:
            all_results[ds] = run_dataset(ds)
        except Exception as e:
            print(f"\n[SKIPPED] {ds}: {e}")

    if len(all_results) > 1:
        print(f"\n{'='*35}")
        print("  MULTI-DATASET MAE SUMMARY — KDEy-ML")
        print(f"{'='*35}")
        print(f"  {'Dataset':<12}  {'MAE':>10}")
        print(f"  {'-'*12}  {'-'*10}")
        for ds, mae in all_results.items():
            v = f"{mae:.5f}" if mae is not None else "ERROR"
            print(f"  {ds:<12}  {v:>10}")
        valid = [v for v in all_results.values() if v is not None]
        print(f"  {'MEAN':<12}  {np.mean(valid):>10.5f}" if valid else "  MEAN  N/A")
        print(f"{'='*35}")


if __name__ == '__main__':
    main()
