"""
Run QuaPy quantification methods on our datasets using exactly the same
train/test sampling protocol as run_ablation.py:
  - synthetic_draw() to create prevalence-controlled train/test splits
  - Loop over dt_distr × train_distr × test_distr × seed × bins configs
  - MAE = mean |estimated_prevalence - actual_test_prevalence| over classes

The QuaPy methods are fit on the train split and quantify on the test split,
without changing the underlying QuaPy method implementations at all.

Usage:
    python run_quapy_comparison.py --dataset beans
    python run_quapy_comparison.py --dataset beans --methods KDEy-ML EMQ
    python run_quapy_comparison.py --all
"""

import sys
import os
import math
import csv
import argparse
import numpy as np

# --- Path setup ---
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(SRC_DIR)
QUAPY_DIR = os.path.join(ROOT_DIR, 'QuaPy')

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, QUAPY_DIR)

import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import PACC, EMQ, DMy
from quapy.method._kdey import KDEyHD, KDEyCS, KDEyML
from sklearn.linear_model import LogisticRegression

import data_loader

SEED = 1

# --------------------------------------------------------------------------
# Method definitions — same methods as ucimulti_experiments.py / commons.py
# --------------------------------------------------------------------------
hyper_LR  = {'classifier__C': np.logspace(-3, 3, 7),
              'classifier__class_weight': ['balanced', None]}
hyper_kde = {'bandwidth': np.linspace(0.01, 0.2, 20)}
nbins     = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20,
             22, 24, 26, 28, 30, 32, 64]

def make_methods(X_leftover=None, y_leftover=None):
    """Return fresh method instances (must be recreated per config).

    val_split=(X_leftover, y_leftover): classifier trains on 100% of the
    synthetic train set (biased prevalence preserved), KDE/aggregation is
    calibrated on the leftover samples (not drawn into train or test).
    Leftover has natural dataset prevalence and is completely disjoint.
    Falls back to val_split=0.4 if no leftover is available.
    """
    val = (X_leftover, y_leftover) if (X_leftover is not None and len(X_leftover) > 0) else 0.4
    return {
        'PACC+':   PACC(LogisticRegression(max_iter=3000, C=1.0), val_split=val),
        'DM-T':    DMy(LogisticRegression(max_iter=3000, C=1.0), val_split=val),
        'DM-HD':   DMy(LogisticRegression(max_iter=3000, C=1.0), val_split=val),
        'DM-CS':   DMy(LogisticRegression(max_iter=3000, C=1.0), val_split=val),
        'EMQ':     EMQ(LogisticRegression(max_iter=3000, C=1.0), val_split=val),
        'KDEy-HD': KDEyHD(LogisticRegression(max_iter=3000, C=1.0), val_split=val),
        'KDEy-CS': KDEyCS(LogisticRegression(max_iter=3000, C=1.0), val_split=val),
        'KDEy-ML': KDEyML(LogisticRegression(max_iter=3000, C=1.0), val_split=val),
    }

# DM method needs divergence set at fit time via set_params
DM_DIVERGENCE = {'DM-T': 'topsoe', 'DM-HD': 'HD', 'DM-CS': 'CS'}

ALL_METHODS = list(make_methods().keys())

# --------------------------------------------------------------------------
# Prevalence tables (identical to run_ablation.py)
# --------------------------------------------------------------------------
train_2_prevalences  = [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]]
train_3_prevalences  = [[0.2, 0.5, 0.3], [0.05, 0.8, 0.15], [0.35, 0.3, 0.35]]
train_4_prevalences  = [[0.5, 0.3, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]]
train_5_prevalences  = [[0.2, 0.15, 0.35, 0.1, 0.2], [0.35, 0.25, 0.15, 0.15, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]]
train_6_prevalences  = [[0.1, 0.2, 0.1, 0.1, 0.25, 0.25], [0.05, 0.1, 0.3, 0.4, 0.1, 0.05], [0.17, 0.17, 0.16, 0.17, 0.17, 0.16]]
train_7_prevalences  = [[0.2, 0.3, 0.2, 0.15, 0.05, 0.05, 0.05], [0.05, 0.1, 0.05, 0.05, 0.25, 0.3, 0.2], [0.15, 0.14, 0.14, 0.15, 0.14, 0.14, 0.14]]
train_10_prevalences = [[0.05, 0.2, 0.05, 0.1, 0.05, 0.25, 0.05, 0.05, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]]

test_2_prevalences  = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5],
                       [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]]
test_3_prevalences  = [[0.1, 0.7, 0.2], [0.55, 0.1, 0.35], [0.35, 0.55, 0.1], [0.4, 0.25, 0.35], [0.0, 0.05, 0.95]]
test_4_prevalences  = [[0.65, 0.25, 0.05, 0.05], [0.2, 0.25, 0.3, 0.25], [0.45, 0.15, 0.2, 0.2], [0.2, 0.0, 0.0, 0.8], [0.3, 0.25, 0.35, 0.1]]
test_5_prevalences  = [[0.15, 0.1, 0.65, 0.1, 0], [0.45, 0.1, 0.3, 0.05, 0.1], [0.2, 0.25, 0.25, 0.1, 0.2], [0.35, 0.05, 0.05, 0.05, 0.5], [0.05, 0.25, 0.15, 0.15, 0.4]]
test_6_prevalences  = [[0.15, 0.1, 0.55, 0.1, 0.0, 0.1], [0.4, 0.1, 0.25, 0.05, 0.1, 0.1], [0.2, 0.2, 0.2, 0.1, 0.2, 0.1], [0.35, 0.05, 0.05, 0.05, 0.05, 0.45], [0.05, 0.25, 0.15, 0.15, 0.1, 0.3]]
test_7_prevalences  = [[0.1, 0.1, 0.1, 0.5, 0.1, 0.0, 0.1], [0.4, 0.1, 0.2, 0.05, 0.1, 0.1, 0.05], [0.15, 0.2, 0.15, 0.1, 0.2, 0.1, 0.1], [0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.45], [0.05, 0.25, 0.1, 0.15, 0.1, 0.3, 0.05]]
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
    constraints = [C] + [y_cts[i] / (dt[0] * train_distr[i] + dt[1] * test_distr[i])
                         for i in range(len(train_distr)) if (dt[0] * train_distr[i] + dt[1] * test_distr[i]) > 0]
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
# Quantify one config with a QuaPy method, return MAE
# --------------------------------------------------------------------------
def quantify_config(method_name, quantifier, X_train, y_train, X_test, y_test,
                    n_classes, test_distr):
    # Set DM divergence if needed
    if method_name in DM_DIVERGENCE:
        quantifier.set_params(divergence=DM_DIVERGENCE[method_name], nbins=10, val_split=10)

    quantifier.fit(X_train, y_train)

    # Quantify: predict class prevalences for the test set
    test_lc = LabelledCollection(X_test, y_test, classes=list(range(n_classes)))
    estim_prev = quantifier.quantify(X_test)

    # Use observed proportions from y_test — matches EDMN's AE() which uses
    # get_actual_count(y_test) / sum(counts), not the configured test_distr.
    # (test_cts are integer-rounded so observed != configured exactly)
    actual_obs = test_lc.prevalence()
    mae = float(np.mean(np.abs(estim_prev - actual_obs)))
    return mae, estim_prev, actual_obs


# --------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------
def run_dataset(dataset, active_methods=None):
    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset}")
    print(f"{'='*70}")

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

    methods_to_run = active_methods if active_methods else ALL_METHODS

    # Accumulate MAEs per method
    all_maes = {m: [] for m in methods_to_run}
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

                    for method_name in methods_to_run:
                        methods = make_methods(X_leftover, y_leftover)
                        quantifier = methods[method_name]
                        try:
                            mae, est, act = quantify_config(
                                method_name, quantifier,
                                X_train, y_train, X_test, y_test,
                                num_classes, test_distr)
                            all_maes[method_name].append(mae)
                        except Exception as e:
                            pass  # skip failed configs silently

                except Exception:
                    skipped += 1
                    continue

    print(f"\n  Configs: {config_id} total, {skipped} skipped\n")
    print(f"  {'Method':<15} {'Mean MAE':>10} {'N configs':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10}")
    for m in methods_to_run:
        vals = all_maes[m]
        if vals:
            print(f"  {m:<15} {np.mean(vals):>10.5f} {len(vals):>10}")
        else:
            print(f"  {m:<15} {'ERROR':>10} {0:>10}")

    return {m: np.mean(v) if v else None for m, v in all_maes.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--all',     action='store_true')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help=f'methods to run. choices: {ALL_METHODS}')
    args = parser.parse_args()

    if args.dataset is None and not args.all:
        parser.error("Provide --dataset <name> or --all")

    datasets = [
        'beans', 'conc', 'drugs', 'yeast', 'gest', 'hars', 'micro',
        'wine', 'turk', 'bike', 'blog', 'diam', 'ener', 'fifa',
        'gasd', 'news', 'nurse', 'optd', 'pend', 'rice',
    ] if args.all else [args.dataset]

    all_results = {}
    for ds in datasets:
        try:
            all_results[ds] = run_dataset(ds, args.methods)
        except Exception as e:
            print(f"\n[SKIPPED] {ds}: {e}")

    if len(all_results) > 1:
        methods_to_show = args.methods if args.methods else ALL_METHODS
        col = 12
        print(f"\n{'='*(12+col*len(methods_to_show)+2)}")
        print("  MULTI-DATASET MAE SUMMARY")
        print(f"{'='*(12+col*len(methods_to_show)+2)}")
        print(f"  {'Dataset':<10}" + "".join(f"{m:>{col}}" for m in methods_to_show))
        print(f"  {'-'*10}" + "".join(f"{'-'*col}" for _ in methods_to_show))
        for ds, res in all_results.items():
            row = f"  {ds:<10}"
            for m in methods_to_show:
                v = res.get(m)
                row += f"{v:>{col}.5f}" if v is not None else f"{'ERROR':>{col}}"
            print(row)
        row = f"  {'MEAN':<10}"
        for m in methods_to_show:
            vals = [res[m] for res in all_results.values() if res.get(m) is not None]
            row += f"{np.mean(vals):>{col}.5f}" if vals else f"{'N/A':>{col}}"
        print(row)
        print(f"{'='*(12+col*len(methods_to_show)+2)}")


if __name__ == '__main__':
    main()
