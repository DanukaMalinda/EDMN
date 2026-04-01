"""
run_ablation_binary_v2.py  —  Ablation study for EDMN on binary datasets (v2)

Uses edm_v2 and enn_binary_v2 instead of the original edm_binary / enn_binary.

Five ablation variants per experimental configuration:
  1. baseline      — plain CE (T=1, no LS, no EDM loss)
  2. label_smooth  — CE + label smoothing  (T=1, LS=0.025, no EDM loss)
  3. temp_scale    — CE + temperature scaling (T=temp, no LS, no EDM loss)
  4. ldm_loss      — CE + EDM loss  (T=1, no LS)
  5. full_edmn     — CE + LS + EDM loss + temperature scaling  (full EDMN)
"""

import os
import math
import copy
import csv
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from tqdm import tqdm

import edm_v2
import enn_binary_v2
import data_loader
import models

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Ablation study — EDMN binary (v2)")
parser.add_argument('--dataset',          type=str,   required=True)
parser.add_argument('--distance_metric',  type=str,   default='HD')
parser.add_argument('--epochs',           type=int,   default=50)
parser.add_argument('--batch_size',       type=int,   default=32)
parser.add_argument('--bins',             type=int,   default=30)
parser.add_argument('--learning_rate',    type=float, default=0.001)
parser.add_argument('--temperature',      type=float, default=2.0,
                    help='Temperature used in variants 3 & 5 during training')
parser.add_argument('--pretrained_dir',   type=str,   default='../models',
                    help='Directory containing pretrained weights '
                         '({dataset}_binary_model_weights_3.pth)')
parser.add_argument('--best_params_json', type=str,   default=None,
                    help='Path to best_params JSON to show tuned MAE in summary')
args = parser.parse_args()


def _load_tuned_mae(dataset):
    """Return (tuned_mae, n_configs_used) from best_params JSON, or (None, None)."""
    if args.best_params_json is None:
        return None, None
    try:
        import json
        with open(args.best_params_json) as f:
            d = json.load(f)
        p = d.get(dataset, {})
        return p.get('achieved_mae'), p.get('tune_epochs')
    except Exception:
        return None, None

device     = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs     = args.epochs
batch_size = args.batch_size
lr         = args.learning_rate
bins       = args.bins
dm         = args.distance_metric
temperature = args.temperature


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class LossWeightLearner(nn.Module):
    """Learned uncertainty-based weighting for two loss terms."""
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

def _get_draw_size(y_cts, dt, train_distr, test_distr, C=None, shared_pool=False):
    if C is None:
        C = sum(y_cts)
    if shared_pool:
        # Each class pool is shared — train and test draw independently,
        # so the binding constraint per class is the larger of the two demands.
        constraints = [C] + [y_cts[i] / max(dt[0] * train_distr[i], dt[1] * test_distr[i])
                              for i in range(len(y_cts))
                              if max(dt[0] * train_distr[i], dt[1] * test_distr[i]) > 0]
    else:
        constraints = [C] + [y_cts[i] / (dt[0] * train_distr[i] + dt[1] * test_distr[i])
                              for i in range(len(y_cts))]
    return np.floor(min(constraints))


def synthetic_draw(n_y, n_classes, y_cts, y_idx,
                   dt_distr, train_distr, test_distr, seed=4711,
                   shared_pool=False):
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

    n          = _get_draw_size(y_cts, dt_distr, train_distr, test_distr,
                                C=n_y, shared_pool=shared_pool)
    train_cts  = (n * dt_distr[0] * train_distr).astype(int)
    if min(train_cts) == 0:
        raise ValueError("Under given distributions a class would be absent from training")
    test_cts   = (n * dt_distr[1] * test_distr).astype(int)

    np.random.seed(seed)
    train_list = [np.random.choice(y_idx[i], size=train_cts[i], replace=False)
                  for i in range(n_classes)]

    if shared_pool:
        # Draw test independently from the same full class pool (overlap allowed)
        test_list = [np.random.choice(y_idx[i], size=test_cts[i], replace=False)
                     for i in range(n_classes)]
    else:
        y_idx_rem = [np.setdiff1d(y_idx[i], train_list[i]) for i in range(n_classes)]
        test_list = [np.random.choice(y_idx_rem[i], size=test_cts[i], replace=False)
                     if len(y_idx_rem[i]) > 0 else []
                     for i in range(n_classes)]

    train_index = np.concatenate(train_list)
    test_index  = np.concatenate(test_list).astype(int)

    n_train = train_index.shape[0]
    n_test  = test_index.shape[0]
    M       = n_train + n_test

    train_ratios = train_cts * 1.0 / n_train
    test_ratios  = test_cts  * 1.0 / n_test
    stats_vec = np.concatenate([
        np.array([M, n_train, n_test, n_train / M, n_test / M]),
        train_cts, train_ratios, test_cts, test_ratios])

    return train_index, test_index, stats_vec


# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------

ABLATION_VARIANTS = [
    {
        'name':              'Baseline NN + EDM',
        'short':             'baseline',
        'label_smoothing':   0.0,
        'train_temperature': 1.0,
        'use_edm_loss':      False,
    },
    {
        'name':              'NN + Label Smoothing + EDM',
        'short':             'label_smooth',
        'label_smoothing':   0.025,
        'train_temperature': 1.0,
        'use_edm_loss':      False,
    },
    {
        'name':              'NN + Temperature Scaling + EDM',
        'short':             'temp_scale',
        'label_smoothing':   0.0,
        'train_temperature': temperature,
        'use_edm_loss':      False,
    },
    {
        'name':              'NN + LDM Loss + EDM',
        'short':             'ldm_loss',
        'label_smoothing':   0.0,
        'train_temperature': 1.0,
        'use_edm_loss':      True,
    },
    {
        'name':              'Full EDMN',
        'short':             'full_edmn',
        'label_smoothing':   0.025,
        'train_temperature': temperature,
        'use_edm_loss':      True,
    },
]


# ---------------------------------------------------------------------------
# Prevalence tables (binary and multiclass)
# ---------------------------------------------------------------------------

_TRAIN = {
    2:  [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]],
    3:  [[0.2, 0.5, 0.3], [0.05, 0.8, 0.15], [0.35, 0.3, 0.35]],
    4:  [[0.5, 0.3, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]],
    5:  [[0.2, 0.15, 0.35, 0.1, 0.2], [0.35, 0.25, 0.15, 0.15, 0.1],
         [0.2, 0.2, 0.2, 0.2, 0.2]],
    6:  [[0.1, 0.2, 0.1, 0.1, 0.25, 0.25], [0.05, 0.1, 0.3, 0.4, 0.1, 0.05],
         [0.17, 0.17, 0.16, 0.17, 0.17, 0.16]],
    7:  [[0.2, 0.3, 0.2, 0.15, 0.05, 0.05, 0.05],
         [0.05, 0.1, 0.05, 0.05, 0.25, 0.3, 0.2],
         [0.15, 0.14, 0.14, 0.15, 0.14, 0.14, 0.14]],
    10: [[0.05, 0.2, 0.05, 0.1, 0.05, 0.25, 0.05, 0.05, 0.1, 0.1],
         [0.15, 0.05, 0.2, 0.05, 0.1, 0.05, 0.2, 0.1, 0.05, 0.05],
         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]],
}

_TEST = {
    2:  [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5],
         [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]],
    3:  [[0.1, 0.7, 0.2], [0.55, 0.1, 0.35], [0.35, 0.55, 0.1],
         [0.4, 0.25, 0.35], [0.0, 0.05, 0.95]],
    4:  [[0.65, 0.25, 0.05, 0.05], [0.2, 0.25, 0.3, 0.25],
         [0.45, 0.15, 0.2, 0.2], [0.2, 0.0, 0.0, 0.8], [0.3, 0.25, 0.35, 0.1]],
    5:  [[0.15, 0.1, 0.65, 0.1, 0], [0.45, 0.1, 0.3, 0.05, 0.1],
         [0.2, 0.25, 0.25, 0.1, 0.2], [0.35, 0.05, 0.05, 0.05, 0.5],
         [0.05, 0.25, 0.15, 0.15, 0.4]],
    6:  [[0.15, 0.1, 0.55, 0.1, 0.0, 0.1], [0.4, 0.1, 0.25, 0.05, 0.1, 0.1],
         [0.2, 0.2, 0.2, 0.1, 0.2, 0.1], [0.35, 0.05, 0.05, 0.05, 0.05, 0.45],
         [0.05, 0.25, 0.15, 0.15, 0.1, 0.3]],
    7:  [[0.1, 0.1, 0.1, 0.5, 0.1, 0.0, 0.1], [0.4, 0.1, 0.2, 0.05, 0.1, 0.1, 0.05],
         [0.15, 0.2, 0.15, 0.1, 0.2, 0.1, 0.1],
         [0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.45],
         [0.05, 0.25, 0.1, 0.15, 0.1, 0.3, 0.05]],
    10: [[0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0, 0.1, 0.05, 0.05],
         [0.2, 0.05, 0.15, 0.05, 0.1, 0.15, 0.05, 0.05, 0.1, 0.1],
         [0, 0.1, 0.05, 0.1, 0.05, 0.1, 0.1, 0.15, 0.15, 0.2],
         [0.05, 0.05, 0.05, 0.35, 0.15, 0.05, 0, 0.1, 0.1, 0.1],
         [0.05, 0.1, 0.1, 0.15, 0.1, 0.15, 0.05, 0.1, 0.1, 0.1]],
}

_DT_RATIOS = [np.array(d) for d in
              [[0.9, 0.1], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3]]]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def collect_predictions(model, loader, device, infer_temperature=1.0):
    """Return (labels, preds, probs) as numpy arrays (eval mode, no_grad)."""
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


def train_variant(train_loader, test_loader, input_dim, num_classes,
                  variant, pretrained_weights=None, init_base_weights=None):
    """
    Train one ablation variant and return the best model (copy).

    Parameters
    ----------
    pretrained_weights : OrderedDict or None
        Weights loaded from ../models/{dataset}_binary_model_weights_3.pth.
        Applied first so every variant starts from the same pretrained base.
    init_base_weights  : OrderedDict or None
        Optional override (e.g. label_smooth weights for full_edmn).
        Applied on top of pretrained_weights when both are provided.
    """
    base_model   = models.getModel(args.dataset, input_dim, num_classes)
    use_edm_loss = variant['use_edm_loss']

    # 1. Start from pretrained weights when available
    if pretrained_weights is not None:
        base_model.load_state_dict(pretrained_weights)

    # 2. Optionally override with weights from a prior variant (e.g. full_edmn
    #    continuing from label_smooth)
    if init_base_weights is not None:
        base_model.load_state_dict(init_base_weights)

    model = FullModel(base_model).to(device) if use_edm_loss else base_model.to(device)

    optimizer  = optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.CrossEntropyLoss(label_smoothing=variant['label_smoothing'])
    train_temp = variant['train_temperature']

    best_model = copy.deepcopy(model)
    min_loss   = np.inf

    for _ in tqdm(range(epochs), desc=f"  [{variant['short']:12s}]", leave=False, disable=True):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits, _ = model(inputs, temperature=train_temp)
            ce_loss   = criterion(logits, labels)

            if use_edm_loss:
                edm_val    = enn_binary_v2.train_edm(
                    train_loader, test_loader, model,
                    device, num_classes, bins, dm)
                edm_tensor = torch.tensor(edm_val, device=device,
                                          dtype=ce_loss.dtype,
                                          requires_grad=True)
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


def run_edm_estimation(model, train_loader, test_loader,
                       num_classes, neighborhood_steps):
    """Run the full EDM estimation pipeline.  Returns (mae, nkld, acc, act, prd, est)."""
    y_train, y_train_pred, probs_train = collect_predictions(
        model, train_loader, device)
    y_test,  y_test_pred,  probs_test  = collect_predictions(
        model, test_loader, device)

    acc     = edm_v2.getAccuracy(y_test, y_test_pred)
    norm_cm = edm_v2.get_coeficient_matrix(y_train, y_train_pred, num_classes)

    ini_estimates = edm_v2.get_init_solution(
        norm_cm, y_test_pred, num_classes, binary=True)

    train_hist_dict, _ = edm_v2.get_train_distributions(
        y_train, y_train_pred, probs_train, num_classes, bins)
    test_hist_dict, _  = edm_v2.get_test_distributions(
        y_test_pred, probs_test, num_classes, bins)

    final_estimation, _ = edm_v2.get_estimation(
        train_hist_dict, test_hist_dict, num_classes,
        neighborhood_steps, ini_estimates, dm)

    act = edm_v2.get_actual_count(y_test, num_classes)
    prd = edm_v2.get_actual_count(y_test_pred, num_classes)
    est = np.sum(final_estimation, axis=0)

    mae  = edm_v2.AE(act, est, num_classes)
    nkld = edm_v2.NKLD(act, est, num_classes)
    return mae, nkld, acc, act, prd, est


# ---------------------------------------------------------------------------
# Summary printers
# ---------------------------------------------------------------------------

SEP  = "=" * 100
HSEP = "-" * 100


def print_config_results(config_id, dt_distr, train_distr, test_distr,
                         seed, bins_val, results):
    print(f"\n  Config #{config_id}  |  dt={np.round(dt_distr,2)}  "
          f"train={np.round(train_distr,2)}  test={np.round(test_distr,2)}  "
          f"seed={seed}  bins={bins_val}")
    print(f"  {'Variant':<35} {'MAE':>8} {'NKLD':>8} {'Acc%':>7}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*7}")
    for v in ABLATION_VARIANTS:
        s = v['short']
        r = results.get(s)
        if r is not None:
            print(f"  {v['name']:<35} {r['mae']:>8.4f} {r['nkld']:>8.4f} {r['acc']:>6.1f}%")
        else:
            print(f"  {v['name']:<35} {'ERROR':>8} {'ERROR':>8} {'ERROR':>7}")


def print_summary(all_results):
    tuned_mae, tuned_epochs = _load_tuned_mae(args.dataset)

    print(f"\n{SEP}")
    print(f"  ABLATION STUDY SUMMARY  |  Dataset: {args.dataset}  |  DM: {dm}")
    print(SEP)
    print(f"  {'Variant':<35} {'Mean MAE':>10} {'Mean NKLD':>10} "
          f"{'Mean Acc%':>10} {'N (acc>=55)':>12}  {'Tuned MAE(30)':>14}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*12}  {'-'*14}")
    for v in ABLATION_VARIANTS:
        s     = v['short']
        valid = [r[s] for r in all_results if s in r and r[s] is not None
                 and r[s]['acc'] >= 55]
        n     = len(valid)
        tuned_col = ''
        if s == 'full_edmn' and tuned_mae is not None:
            tuned_col = f'  {tuned_mae:>14.4f}'
        if n > 0:
            print(f"  {v['name']:<35} {np.mean([r['mae']  for r in valid]):>10.4f} "
                  f"{np.mean([r['nkld'] for r in valid]):>10.4f} "
                  f"{np.mean([r['acc']  for r in valid]):>9.1f}% {n:>12}{tuned_col}")
        else:
            print(f"  {v['name']:<35} {'N/A':>10} {'N/A':>10} {'N/A':>10} {0:>12}{tuned_col}")
    print(SEP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_pretrained(dataset, input_dim, num_classes):
    """
    Try to load pretrained weights for *dataset* from the pretrained_dir.

    Looks for  {pretrained_dir}/{dataset}_binary_model_weights_3.pth
    Returns the state-dict (OrderedDict) on success, or None if the file
    does not exist or cannot be loaded.
    """
    path = os.path.join(args.pretrained_dir,
                        f'{dataset}_binary_model_weights_3.pth')
    if not os.path.isfile(path):
        return None

    try:
        probe = models.getModel(dataset, input_dim, num_classes)
        state = torch.load(path, map_location='cpu')
        probe.load_state_dict(state)
        return state
    except Exception:
        return None


def main():
    X_n, y_n = data_loader.load_data(args.dataset)

    Y           = np.unique(y_n)
    num_classes = len(Y)
    y_idx       = [np.where(y_n == l)[0] for l in Y]
    y_cts       = np.array([len(np.where(y_n == l)[0]) for l in Y])
    N           = len(X_n)

    if num_classes not in _TRAIN:
        raise ValueError(f"No prevalence table for {num_classes} classes")

    train_ds = np.array(_TRAIN[num_classes])
    test_ds  = np.array(_TEST[num_classes])

    # Pre-compute EDM neighbourhood steps
    neighborhood_steps = np.array(list(edm_v2.get_steps(num_classes)))
    for i in range(2):
        extra = edm_v2.get_additional_neighbors(num_classes, step=(i + 2) * 2)
        neighborhood_steps = np.vstack([neighborhood_steps, extra])

    # Output CSV
    out_dir  = '../results_ablation_binary'
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f'{out_dir}/ablation_binary_{args.dataset}_{dm}_v2.csv'

    csv_headers = ['config_id', 'dt_distr', 'train_distr', 'test_distr', 'seed', 'bins']
    for v in ABLATION_VARIANTS:
        s = v['short']
        csv_headers += [f'mae_{s}', f'nkld_{s}', f'acc_{s}']
    csv_headers += ['act', 'prd']

    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(csv_headers)

    seed_list   = [11]
    bin_list    = [bins]
    shared_pool = N < 5000   # small datasets: train+test draw from the same class pool

    all_results = []
    config_id   = 0

    for dt_distr in _DT_RATIOS:
        for train_distr in train_ds:
            for test_distr in test_ds:
                for seed in seed_list:
                    for bins_val in bin_list:
                        config_id += 1
                        config_results = {}

                        try:
                            train_index, test_index, _ = synthetic_draw(
                                N, num_classes, y_cts, y_idx,
                                dt_distr, train_distr, test_distr, seed,
                                shared_pool=shared_pool)

                            X_train, y_train = X_n[train_index], y_n[train_index]
                            X_test,  y_test  = X_n[test_index],  y_n[test_index]

                            if len(X_test) < 100:
                                continue

                            input_dim        = X_train.shape[1]

                            # Dynamic batch size: ensure at least 4 batches so
                            # that training is not trivially a single-step update.
                            # Never go below 8 or above the requested batch_size.
                            eff_bs = min(batch_size,
                                         max(8, len(X_train) // 4))

                            train_loader = DataLoader(
                                TensorDataset(
                                    torch.tensor(X_train, dtype=torch.float32),
                                    torch.tensor(y_train, dtype=torch.long)),
                                batch_size=eff_bs, shuffle=True)
                            test_loader = DataLoader(
                                TensorDataset(
                                    torch.tensor(X_test, dtype=torch.float32),
                                    torch.tensor(y_test, dtype=torch.long)),
                                batch_size=eff_bs)

                            # Load pretrained weights once; all variants finetune from here
                            pretrained_weights = _load_pretrained(
                                args.dataset, input_dim, num_classes)

                            saved_base_weights = {}

                            for variant in ABLATION_VARIANTS:
                                short = variant['short']
                                try:
                                    # full_edmn continues from label_smooth weights
                                    # (which themselves were finetuned from pretrained)
                                    init_w = (saved_base_weights.get('label_smooth')
                                              if short == 'full_edmn' else None)

                                    model = train_variant(
                                        train_loader, test_loader,
                                        input_dim, num_classes, variant,
                                        pretrained_weights=pretrained_weights,
                                        init_base_weights=init_w)

                                    base = model.mlp if isinstance(model, FullModel) else model
                                    saved_base_weights[short] = copy.deepcopy(
                                        base.state_dict())

                                    mae, nkld, acc, act, prd, est = run_edm_estimation(
                                        model, train_loader, test_loader,
                                        num_classes, neighborhood_steps)

                                    config_results[short] = {
                                        'mae': mae, 'nkld': nkld, 'acc': acc,
                                        'act': act, 'prd': prd, 'est': est,
                                    }

                                except Exception as e:
                                    config_results[short] = None

                            all_results.append(config_results)

                            # Write one CSV row per configuration
                            row = [config_id,
                                   list(np.round(dt_distr, 3)),
                                   list(np.round(train_distr, 3)),
                                   list(np.round(test_distr, 3)),
                                   seed, bins_val]
                            for v in ABLATION_VARIANTS:
                                r = config_results.get(v['short'])
                                row += [r['mae'], r['nkld'], r['acc']] if r else [None, None, None]
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
