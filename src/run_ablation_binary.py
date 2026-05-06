"""
Ablation study for EDMN on binary datasets.
Runs 5 variants in sequence for each experimental configuration:
  1. Baseline NN + EDM          (plain CE, T=1, no EDM loss)
  2. NN + Label Smoothing + EDM (CE with LS=0.025, T=1, no EDM loss)
  3. NN + Temperature Scaling + EDM (plain CE, T=temp during training, no EDM loss)
  4. NN + LDM Loss + EDM        (CE + EDM loss, T=1, no LS)
  5. Full EDMN                   (CE+LS + EDM loss + T=temp)
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import edm_binary
import enn_binary
import csv
import argparse
import data_loader
import models
import math
import os
import copy
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Ablation study for EDMN (binary datasets)")
parser.add_argument('--dataset',          type=str,   required=True)
parser.add_argument('--distance_metric',  type=str,   default='HD')
parser.add_argument('--epochs',           type=int,   default=50)
parser.add_argument('--batch_size',       type=int,   default=32)
parser.add_argument('--bins',             type=int,   default=30)
parser.add_argument('--learning_rate',    type=float, default=0.001)
parser.add_argument('--temperature',      type=float, default=2.0,
                    help='Temperature used in variants 3 & 5 during training')
args = parser.parse_args()

epochs      = args.epochs
batch_size  = args.batch_size
lr          = args.learning_rate
bins        = args.bins
dm          = args.distance_metric
temperature = args.temperature

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class LossWeightLearner(nn.Module):
    def __init__(self, init_val=0.0, clamp_range=(-10, 10)):
        super().__init__()
        self.log_sigma1  = nn.Parameter(torch.tensor([init_val], dtype=torch.float))
        self.log_sigma2  = nn.Parameter(torch.tensor([init_val], dtype=torch.float))
        self.clamp_range = clamp_range

    def forward(self, loss1, loss2):
        s1 = torch.clamp(self.log_sigma1, *self.clamp_range)
        s2 = torch.clamp(self.log_sigma2, *self.clamp_range)
        return (torch.exp(-s1) * loss1 + s1) + (torch.exp(-s2) * loss2 + s2)


class FullModel(nn.Module):
    """Wraps a base model and adds learned loss weighting for CE + EDM."""
    def __init__(self, base_model):
        super().__init__()
        self.mlp          = base_model
        self.loss_weights = LossWeightLearner(init_val=0.5)

    def forward(self, x, temperature=1.0):
        return self.mlp(x, temperature)


# ---------------------------------------------------------------------------
# Data split utilities
# ---------------------------------------------------------------------------
def get_draw_size(y_cts, dt, train_distr, test_distr, C=None):
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

    np.random.seed(seed)
    train_list = [np.random.choice(y_idx[i], size=train_cts[i], replace=False)
                  for i in range(n_classes)]
    y_idx_rem  = [np.setdiff1d(y_idx[i], train_list[i]) for i in range(n_classes)]
    test_list  = [np.random.choice(y_idx_rem[i], size=test_cts[i], replace=False)
                  if len(y_idx_rem[i]) > 0 else []
                  for i in range(n_classes)]

    train_index = np.concatenate(train_list)
    test_index  = np.concatenate(test_list).astype(int)

    n_train = train_index.shape[0]
    n_test  = test_index.shape[0]
    M       = n_train + n_test
    r_train = n_train * 1.0 / M
    r_test  = n_test  * 1.0 / M

    train_ratios = train_cts * 1.0 / n_train
    test_ratios  = test_cts  * 1.0 / n_test

    stats_vec = np.concatenate(
        [np.array([M, n_train, n_test, r_train, r_test]),
         train_cts, train_ratios, test_cts, test_ratios])

    return train_index, test_index, stats_vec


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def collect_predictions(model, loader, device, infer_temperature=1.0):
    """Return (labels, predicted_classes, max_probs) as numpy arrays."""
    labels_all, preds_all, probs_all = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, probs = model(inputs, infer_temperature)
            max_probs, predicted = torch.max(probs, dim=1)
            labels_all.append(labels.cpu())
            preds_all.append(predicted.cpu())
            probs_all.append(max_probs.cpu())
    return (torch.cat(labels_all).numpy(),
            torch.cat(preds_all).numpy(),
            torch.cat(probs_all).numpy())


def train_variant(train_loader, test_loader, input_dim, num_classes, variant,
                  init_base_weights=None):
    """Train one ablation variant and return the best model."""
    base_model   = models.getModel(args.dataset, input_dim, num_classes)
    use_edm_loss = variant['use_edm_loss']

    if init_base_weights is not None:
        base_model.load_state_dict(init_base_weights)

    if use_edm_loss:
        model = FullModel(base_model).to(device)
    else:
        model = base_model.to(device)

    optimizer  = optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.CrossEntropyLoss(label_smoothing=variant['label_smoothing'])
    train_temp = variant['train_temperature']

    best_model = copy.deepcopy(model)
    min_loss   = np.inf

    for _ in tqdm(range(epochs), desc=f"  [{variant['short']:12s}]", leave=False):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits, _ = model(inputs, temperature=train_temp)
            ce_loss   = criterion(logits, labels)

            if use_edm_loss:
                edm_val    = enn_binary.train_edm(train_loader, test_loader, model, device,
                                                  num_classes, bins, dm)
                edm_tensor = torch.tensor(edm_val, device=device,
                                          dtype=ce_loss.dtype, requires_grad=True)
                loss = model.loss_weights(ce_loss, edm_tensor)
            else:
                loss = ce_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if avg_loss < min_loss:
            min_loss   = avg_loss
            best_model = copy.deepcopy(model)

    return best_model


def run_edm_estimation(model, train_loader, test_loader, num_classes, neighborhood_steps):
    """Run the full EDM estimation pipeline. Returns (mae, nkld, acc, act, prd, est)."""
    y_train, y_train_pred, probs_train = collect_predictions(model, train_loader, device, 1.0)
    y_test,  y_test_pred,  probs_test  = collect_predictions(model, test_loader,  device, 1.0)

    acc     = edm_binary.getAccuracy(y_test, y_test_pred)
    norm_cm = edm_binary.get_coeficient_matrix(y_train, y_train_pred, num_classes)

    # Returns a list of initial estimates (binary-specific)
    ini_estimates = edm_binary.get_init_solution(norm_cm, y_test_pred, num_classes)

    train_hist_dict, _ = edm_binary.get_train_distributions(
        y_train, y_train_pred, probs_train, num_classes, bins)
    test_hist_dict, _  = edm_binary.get_test_distributions(
        y_test_pred, probs_test, num_classes, bins)

    final_estimation, _ = edm_binary.get_estimation(
        train_hist_dict, test_hist_dict, num_classes,
        neighborhood_steps, ini_estimates, dm)

    act = edm_binary.get_actual_count(y_test, num_classes)
    prd = edm_binary.get_actual_count(y_test_pred, num_classes)
    est = np.sum(final_estimation, axis=0)

    mae  = edm_binary.AE(act, est, num_classes)
    nkld = edm_binary.NKLD(act, est, num_classes)
    return mae, nkld, acc, act, prd, est


# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------
ABLATION_VARIANTS = [
    {
        'name':             'Baseline NN + EDM',
        'short':            'baseline',
        'label_smoothing':  0.0,
        'train_temperature': 1.0,
        'use_edm_loss':     False,
    },
    {
        'name':             'NN + Label Smoothing + EDM',
        'short':            'label_smooth',
        'label_smoothing':  0.025,
        'train_temperature': 1.0,
        'use_edm_loss':     False,
    },
    {
        'name':             'NN + Temperature Scaling + EDM',
        'short':            'temp_scale',
        'label_smoothing':  0.0,
        'train_temperature': temperature,
        'use_edm_loss':     False,
    },
    {
        'name':             'NN + LDM Loss + EDM',
        'short':            'ldm_loss',
        'label_smoothing':  0.0,
        'train_temperature': 1.0,
        'use_edm_loss':     True,
    },
    {
        'name':             'Full EDMN',
        'short':            'full_edmn',
        'label_smoothing':  0.025,
        'train_temperature': temperature,
        'use_edm_loss':     True,
    },
]

# ---------------------------------------------------------------------------
# Prevalence tables (binary datasets)
# ---------------------------------------------------------------------------
train_2_prevalences  = [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]]
train_3_prevalences  = [[0.2, 0.5, 0.3], [0.05, 0.8, 0.15], [0.35, 0.3, 0.35]]
train_4_prevalences  = [[0.5, 0.3, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]]
train_5_prevalences  = [[0.2, 0.15, 0.35, 0.1, 0.2], [0.35, 0.25, 0.15, 0.15, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]]
train_6_prevalences  = [[0.1, 0.2, 0.1, 0.1, 0.25, 0.25], [0.05, 0.1, 0.3, 0.4, 0.1, 0.05], [0.17, 0.17, 0.16, 0.17, 0.17, 0.16]]
train_7_prevalences  = [[0.2, 0.3, 0.2, 0.15, 0.05, 0.05, 0.05], [0.05, 0.1, 0.05, 0.05, 0.25, 0.3, 0.2], [0.15, 0.14, 0.14, 0.15, 0.14, 0.14, 0.14]]
train_10_prevalences = [[0.05, 0.2, 0.05, 0.1, 0.05, 0.25, 0.05, 0.05, 0.1, 0.1],
                        [0.15, 0.05, 0.2, 0.05, 0.1, 0.05, 0.2, 0.1, 0.05, 0.05],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]

test_2_prevalences  = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5],
                        [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]]
test_3_prevalences  = [[0.1, 0.7, 0.2], [0.55, 0.1, 0.35], [0.35, 0.55, 0.1], [0.4, 0.25, 0.35], [0.0, 0.05, 0.95]]
test_4_prevalences  = [[0.65, 0.25, 0.05, 0.05], [0.2, 0.25, 0.3, 0.25], [0.45, 0.15, 0.2, 0.2], [0.2, 0.0, 0.0, 0.8], [0.3, 0.25, 0.35, 0.1]]
test_5_prevalences  = [[0.15, 0.1, 0.65, 0.1, 0], [0.45, 0.1, 0.3, 0.05, 0.1], [0.2, 0.25, 0.25, 0.1, 0.2], [0.35, 0.05, 0.05, 0.05, 0.5], [0.05, 0.25, 0.15, 0.15, 0.4]]
test_6_prevalences  = [[0.15, 0.1, 0.55, 0.1, 0.0, 0.1], [0.4, 0.1, 0.25, 0.05, 0.1, 0.1], [0.2, 0.2, 0.2, 0.1, 0.2, 0.1], [0.35, 0.05, 0.05, 0.05, 0.05, 0.45], [0.05, 0.25, 0.15, 0.15, 0.1, 0.3]]
test_7_prevalences  = [[0.1, 0.1, 0.1, 0.5, 0.1, 0.0, 0.1], [0.4, 0.1, 0.2, 0.05, 0.1, 0.1, 0.05], [0.15, 0.2, 0.15, 0.1, 0.2, 0.1, 0.1], [0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.45], [0.05, 0.25, 0.1, 0.15, 0.1, 0.3, 0.05]]
test_10_prevalences = [[0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0, 0.1, 0.05, 0.05],
                       [0.2, 0.05, 0.15, 0.05, 0.1, 0.15, 0.05, 0.05, 0.1, 0.1],
                       [0, 0.1, 0.05, 0.1, 0.05, 0.1, 0.1, 0.15, 0.15, 0.2],
                       [0.05, 0.05, 0.05, 0.35, 0.15, 0.05, 0, 0.1, 0.1, 0.1],
                       [0.05, 0.1, 0.1, 0.15, 0.1, 0.15, 0.05, 0.1, 0.1, 0.1]]

train_prev_dictionary = {2: train_2_prevalences, 3: train_3_prevalences,
                         4: train_4_prevalences, 5: train_5_prevalences,
                         6: train_6_prevalences, 7: train_7_prevalences,
                         10: train_10_prevalences}
test_prev_dictionary  = {2: test_2_prevalences,  3: test_3_prevalences,
                         4: test_4_prevalences,  5: test_5_prevalences,
                         6: test_6_prevalences,  7: test_7_prevalences,
                         10: test_10_prevalences}

train_test_ratios = [np.array(d) for d in [[0.9, 0.1], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3]]]


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------
SEP  = "=" * 100
HSEP = "-" * 100

def print_config_results(config_id, dt_distr, train_distr, test_distr, seed, bins_val, results):
    print(f"\n  Config #{config_id}  |  dt={np.round(dt_distr,2)}  "
          f"train={np.round(train_distr,2)}  test={np.round(test_distr,2)}  "
          f"seed={seed}  bins={bins_val}")
    print(f"  {'Variant':<35} {'NKLD':>8} {'Acc%':>7}")
    print(f"  {'-'*35} {'-'*8} {'-'*7}")
    for v in ABLATION_VARIANTS:
        s = v['short']
        if s in results and results[s] is not None:
            r = results[s]
            print(f"  {v['name']:<35} {r['nkld']:>8.4f} {r['acc']:>6.1f}%")
        else:
            print(f"  {v['name']:<35} {'ERROR':>8} {'ERROR':>7}")


def print_summary(all_results):
    print(f"\n{SEP}")
    print(f"  ABLATION STUDY SUMMARY  |  Dataset: {args.dataset}  |  DM: {dm}")
    print(SEP)
    print(f"  {'Variant':<35} {'Mean MAE':>10} {'Mean NKLD':>10} {'Mean Acc%':>10} {'N':>5}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*5}")
    for v in ABLATION_VARIANTS:
        s     = v['short']
        maes  = [r[s]['mae']  for r in all_results if s in r and r[s] is not None and r[s]['acc'] >= 55]
        nklds = [r[s]['nkld'] for r in all_results if s in r and r[s] is not None and r[s]['acc'] >= 55]
        accs  = [r[s]['acc']  for r in all_results if s in r and r[s] is not None and r[s]['acc'] >= 55]
        n     = len(maes)
        if n > 0:
            print(f"  {v['name']:<35} {np.mean(maes):>10.4f} {np.mean(nklds):>10.4f} "
                  f"{np.mean(accs):>9.1f}% {n:>5}")
        else:
            print(f"  {v['name']:<35} {'N/A':>10} {'N/A':>10} {'N/A':>10} {0:>5}")
    print(SEP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    X_n, y_n = data_loader.load_data(args.dataset)

    Y           = np.unique(y_n)
    num_classes = len(Y)
    y_idx       = [np.where(y_n == l)[0] for l in Y]
    y_cts       = np.array([len(np.where(y_n == l)[0]) for l in Y])
    N           = len(X_n)

    if num_classes not in train_prev_dictionary:
        raise ValueError(f"No prevalence table for {num_classes} classes")

    train_ds = np.array(train_prev_dictionary[num_classes])
    test_ds  = np.array(test_prev_dictionary[num_classes])

    # Pre-compute EDM neighbourhood steps (binary style)
    neighborhood_steps = np.array(list(edm_binary.get_steps(num_classes)))
    for i in range(2):
        additional = edm_binary.get_additional_neighbors(num_classes, step=(i + 2) * 2)
        neighborhood_steps = np.vstack([neighborhood_steps, additional])

    # Output CSV
    out_dir = '../results_ablation_binary'
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f'{out_dir}/ablation_binary_{args.dataset}_{dm}.csv'

    csv_headers = ['config_id', 'dt_distr', 'train_distr', 'test_distr', 'seed', 'bins']
    for v in ABLATION_VARIANTS:
        s = v['short']
        csv_headers += [f'mae_{s}', f'nkld_{s}', f'acc_{s}']
    csv_headers += ['act', 'prd']

    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(csv_headers)

    seed_list = [11]
    bin_list  = [30]

    all_results = []
    config_id   = 0

    for dt_distr in train_test_ratios:
        for train_distr in train_ds:
            for test_distr in test_ds:
                for seed in seed_list:
                    for bins_val in bin_list:
                        config_id += 1
                        config_results     = {}
                        saved_base_weights = {}

                        try:
                            train_index, test_index, _ = synthetic_draw(
                                N, num_classes, y_cts, y_idx,
                                dt_distr, train_distr, test_distr, seed)

                            X_train, y_train = X_n[train_index], y_n[train_index]
                            X_test,  y_test  = X_n[test_index],  y_n[test_index]
                            input_dim        = X_train.shape[1]

                            X_train_t = torch.tensor(X_train, dtype=torch.float32)
                            y_train_t = torch.tensor(y_train, dtype=torch.long)
                            X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
                            y_test_t  = torch.tensor(y_test,  dtype=torch.long)

                            train_loader = DataLoader(
                                TensorDataset(X_train_t, y_train_t),
                                batch_size=batch_size, shuffle=True)
                            test_loader  = DataLoader(
                                TensorDataset(X_test_t, y_test_t),
                                batch_size=batch_size)

                            # ---- Run each ablation variant ----
                            for variant in ABLATION_VARIANTS:
                                short = variant['short']
                                try:
                                    init_weights = (saved_base_weights.get('label_smooth')
                                                    if short == 'full_edmn' else None)

                                    model = train_variant(
                                        train_loader, test_loader,
                                        input_dim, num_classes, variant,
                                        init_base_weights=init_weights)

                                    base = model.mlp if isinstance(model, FullModel) else model
                                    saved_base_weights[short] = copy.deepcopy(base.state_dict())

                                    mae, nkld, acc, act, prd, est = run_edm_estimation(
                                        model, train_loader, test_loader,
                                        num_classes, neighborhood_steps)

                                    config_results[short] = {
                                        'mae':  mae,
                                        'nkld': nkld,
                                        'acc':  acc,
                                        'act':  act,
                                        'prd':  prd,
                                        'est':  est,
                                    }

                                except Exception as e:
                                    config_results[short] = None

                            all_results.append(config_results)

                            # Write CSV row
                            row = [config_id,
                                   list(np.round(dt_distr, 3)),
                                   list(np.round(train_distr, 3)),
                                   list(np.round(test_distr, 3)),
                                   seed, bins_val]
                            for v in ABLATION_VARIANTS:
                                s = v['short']
                                r = config_results.get(s)
                                if r:
                                    row += [r['mae'], r['nkld'], r['acc']]
                                else:
                                    row += [None, None, None]
                            fe = config_results.get('full_edmn')
                            row += [list(fe['act']) if fe else None,
                                    list(fe['prd']) if fe else None]

                            with open(csv_path, 'a', newline='') as f:
                                csv.writer(f).writerow(row)

                        except Exception as e:
                            continue

    print_summary(all_results)


if __name__ == '__main__':
    main()
