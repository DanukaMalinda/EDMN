"""
tune_edmn_binary.py — Find best (temperature, bins, epsilon) for EDMN binary.

Performs a grid search over:
    temperature  T ∈ {1.5, 2.0, 2.5, 3.0}        (softmax spread)
    bins         b ∈ {30, 60, 90}                  (histogram resolution)
    epsilon      e ∈ {0.0, 0.01, 0.025, 0.05}      (label smoothing)

For each combination, trains one model and measures EDM validation MAE.
The best (T, bins, epsilon) per dataset is saved to best_params_binary.json.

Usage:
    # Tune a single dataset:
    python tune_edmn_binary.py --dataset cappl

    # Tune multiple datasets:
    python tune_edmn_binary.py --dataset cappl bc_cont wine_b

    # Tune ALL 36 binary datasets:
    python tune_edmn_binary.py --all

    # Quick scan (fewer epochs per combo):
    python tune_edmn_binary.py --all --epochs 40 --patience 10 --warmup 8

Output:
    ../results_ablation_binary/best_params_binary.json
    Keys per dataset: {"temperature": T, "bins": b, "epsilon": e}
"""

import os, copy, json, argparse, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_loader, models, edm_binary

# ---------------------------------------------------------------------------
# All 36 binary datasets (paper Table 5)
# ---------------------------------------------------------------------------
ALL_DATASETS = [
    'cappl',   'bc_cont', 'study_b', 'alco',   'flare_b', 'contra_b',
    'cars_b',  'drugs_b', 'ads',     'craft_b', 'spam_b',  'turk_b',
    'thrm_b',  'wine_b',  'musk',    'vgame_b', 'mush',    'grid_b',
    'yeast_b', 'phish',   'fifa_b',  'bike_b',  'magic',   'ener_b',
    'avila',   'cond_b',  'adult',   'diam_b',  'blog_b',  'dota_b',
    'boone',   'conc_b',  'music',   'news_b',  'occup',   'nurse_b',
]

# Search grid
TEMP_CANDIDATES    = [1.5, 2.0, 2.5, 3.0]
BINS_CANDIDATES    = [30, 60, 90]
EPSILON_CANDIDATES = [0.0, 0.01, 0.025, 0.05]

# Fixed EDM test prevalences
TEST_PREVS = [
    [0.05, 0.95], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6],
    [0.5, 0.5],   [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1],
    [0.95, 0.05],
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Tune EDMN hyperparameters per dataset')
parser.add_argument('--dataset',    type=str, nargs='+', default=[])
parser.add_argument('--all',        action='store_true', help='Tune all 36 datasets')
parser.add_argument('--output',     type=str,
                    default='../results_ablation_binary/best_params_binary.json')
parser.add_argument('--epochs',     type=int, default=80,
                    help='Training epochs per combo (default 80)')
parser.add_argument('--patience',   type=int, default=15,
                    help='Early stopping patience on EDM val loss')
parser.add_argument('--warmup',     type=int, default=15,
                    help='Warmup epochs before early stopping activates')
parser.add_argument('--lr',         type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dm',         type=str, default='HD',
                    help='Distance metric: HD, SE, MH, TS, etc.')
parser.add_argument('--seed',       type=int, default=42)
# Narrow the grid at the CLI if you want a faster run:
parser.add_argument('--temps',      type=float, nargs='+', default=TEMP_CANDIDATES)
parser.add_argument('--bins_list',  type=int,   nargs='+', default=BINS_CANDIDATES)
parser.add_argument('--epsilons',   type=float, nargs='+', default=EPSILON_CANDIDATES)
args = parser.parse_args()

if args.all:
    datasets_to_tune = ALL_DATASETS
elif args.dataset:
    datasets_to_tune = args.dataset
else:
    parser.error('Specify --dataset <name> [<name> ...] or --all')

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

grid = list(itertools.product(args.temps, args.bins_list, args.epsilons))
print(f'Grid size: {len(grid)} combos  '
      f'(T={args.temps}  bins={args.bins_list}  eps={args.epsilons})\n')

# ---------------------------------------------------------------------------
# Helper: collect model predictions from a DataLoader
# ---------------------------------------------------------------------------
def collect_preds(model, loader, temperature):
    model.eval()
    ys, preds, probs_list = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            logits, _ = model(Xb.to(device))
            p = torch.softmax(logits / temperature, dim=1).cpu().numpy()
            probs_list.append(p)
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            ys.extend(yb.numpy().tolist())
    return np.array(ys), np.array(preds), np.vstack(probs_list)


# ---------------------------------------------------------------------------
# EDM validation MAE: sweep synthetic test prevalences, return mean MAE
# ---------------------------------------------------------------------------
def edm_val_mae(model, X, y, num_classes, temperature, bins, dm):
    classes, counts = np.unique(y, return_counts=True)
    minority = counts.min()
    ctx_per  = max(15, int(0.20 * minority))
    ctx_per  = min(ctx_per, minority // 2)

    # Reserve balanced context
    ctx_idx = []
    for c in classes:
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        ctx_idx.extend(idx[:ctx_per].tolist())
    ctx_idx = np.array(ctx_idx)

    ctx_loader = DataLoader(
        TensorDataset(torch.tensor(X[ctx_idx], dtype=torch.float32),
                      torch.tensor(y[ctx_idx], dtype=torch.long)),
        batch_size=128)
    y_ctx, y_ctx_pred, probs_ctx = collect_preds(model, ctx_loader, temperature)

    total_mae, n_valid = 0.0, 0

    for prev in TEST_PREVS:
        n_test = min(200, int(minority / max(prev[0], prev[1])))
        if n_test < 20:
            continue
        n0 = max(1, int(n_test * prev[0]))
        n1 = max(1, int(n_test * prev[1]))
        idx0 = np.setdiff1d(np.where(y == 0)[0], ctx_idx)
        idx1 = np.setdiff1d(np.where(y == 1)[0], ctx_idx)
        if len(idx0) < n0 or len(idx1) < n1:
            continue
        te_idx = np.concatenate([
            np.random.choice(idx0, n0, replace=False),
            np.random.choice(idx1, n1, replace=False),
        ])
        te_loader = DataLoader(
            TensorDataset(torch.tensor(X[te_idx], dtype=torch.float32),
                          torch.tensor(y[te_idx], dtype=torch.long)),
            batch_size=128)
        y_te, y_te_pred, probs_te = collect_preds(model, te_loader, temperature)

        try:
            tr_hist, _ = edm_binary.get_train_distributions(
                y_ctx, y_ctx_pred, probs_ctx, num_classes, bins)
            te_hist, _ = edm_binary.get_test_distributions(
                y_te_pred, probs_te, num_classes, bins)
            norm_cm  = edm_binary.get_coeficient_matrix(y_ctx, y_ctx_pred, num_classes)
            ini_est  = edm_binary.get_init_solution(norm_cm, y_te_pred, num_classes)
            nb_steps = np.array(list(edm_binary.get_steps(num_classes)))
            est, _   = edm_binary.get_estimation(
                tr_hist, te_hist, num_classes, nb_steps, ini_est, dm)
            act      = edm_binary.get_actual_count(y_te, num_classes)
            total_mae += edm_binary.AE(act, est, num_classes)
            n_valid   += 1
        except Exception:
            continue

    return total_mae / n_valid if n_valid > 0 else float('inf')


# ---------------------------------------------------------------------------
# Train one model with given (T, bins, epsilon) and return best EDM val MAE
# ---------------------------------------------------------------------------
def train_and_eval(X, y, num_classes, dataset_name, temperature, bins, epsilon):
    input_dim    = X.shape[1]
    class_counts = np.bincount(y, minlength=num_classes).astype(float)
    sample_w     = np.array([1.0 / class_counts[yi] for yi in y])
    sampler      = WeightedRandomSampler(
        torch.tensor(sample_w, dtype=torch.float32), len(y), replacement=True)
    pos_w = torch.tensor(
        [class_counts.mean() / (class_counts[c] + 1e-8) for c in range(num_classes)],
        dtype=torch.float32).to(device)

    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.long)),
        batch_size=args.batch_size, sampler=sampler, drop_last=True)

    model     = models.getModel(dataset_name, input_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=pos_w, label_smoothing=epsilon)

    best_mae   = float('inf')
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits, _ = model(Xb)
            loss = criterion(logits / temperature, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch >= args.warmup:
            mae = edm_val_mae(model, X, y, num_classes, temperature, bins, args.dm)
            if mae < best_mae:
                best_mae   = mae
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    break

    return best_mae


# ---------------------------------------------------------------------------
# Load / update JSON
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(args.output))
os.makedirs(out_dir, exist_ok=True)
if os.path.exists(args.output):
    with open(args.output) as f:
        best_params = json.load(f)
    print(f'Loaded existing {args.output} ({len(best_params)} entries)\n')
else:
    best_params = {}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print(f'Tuning {len(datasets_to_tune)} dataset(s)')
print(f'Total combos per dataset: {len(grid)}\n')

for dataset in datasets_to_tune:
    print(f'{"="*60}')
    print(f'  Dataset: {dataset}')
    print(f'{"="*60}')

    try:
        X, y = data_loader.load_data(dataset)
    except Exception as e:
        print(f'  ERROR loading data: {e}\n')
        continue

    num_classes = len(np.unique(y))
    print(f'  N={len(X)}  D={X.shape[1]}  classes={num_classes}')

    if num_classes != 2:
        print(f'  SKIP — not binary ({num_classes} classes)\n')
        continue

    results = {}
    best_mae_so_far = float('inf')

    for i, (T, b, e) in enumerate(grid, 1):
        print(f'  [{i:>2}/{len(grid)}] T={T}  bins={b}  eps={e} ... ', end='', flush=True)
        mae = train_and_eval(X, y, num_classes, dataset, T, b, e)
        results[(T, b, e)] = mae
        marker = ' <-- best' if mae < best_mae_so_far else ''
        print(f'MAE={mae:.4f}{marker}')
        if mae < best_mae_so_far:
            best_mae_so_far = mae

    best_combo = min(results, key=lambda k: results[k])
    best_T, best_b, best_e = best_combo
    best_mae = results[best_combo]

    print(f'\n  => Best: T={best_T}  bins={best_b}  eps={best_e}  MAE={best_mae:.4f}')

    # Show top-5 combos
    ranked = sorted(results.items(), key=lambda x: x[1])[:5]
    print(f'  Top 5:')
    for (T, b, e), mae in ranked:
        print(f'    T={T}  bins={b:>2}  eps={e:.3f}  MAE={mae:.4f}')

    best_params[dataset] = {
        'temperature': best_T,
        'bins':        best_b,
        'epsilon':     best_e,
        'edm_val_mae': round(best_mae, 6),
    }

    with open(args.output, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f'  Saved -> {args.output}\n')

print('\nAll done.')
print(f'\n{"Dataset":<14}  {"T":>5}  {"bins":>4}  {"eps":>6}  {"MAE":>8}')
print('-' * 44)
for ds in datasets_to_tune:
    if ds in best_params:
        p = best_params[ds]
        print(f'{ds:<14}  {p["temperature"]:>5}  {p["bins"]:>4}  '
              f'{p["epsilon"]:>6}  {p["edm_val_mae"]:>8.4f}')
