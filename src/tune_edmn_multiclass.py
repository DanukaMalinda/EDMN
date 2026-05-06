"""
tune_edmn_multiclass.py — Find best (temperature, bins, epsilon) for EDMN multiclass.

Grid search over:
    temperature  T ∈ {1.5, 2.0, 2.5, 3.0}
    bins         b ∈ {30, 60, 90}
    epsilon      e ∈ {0.0, 0.01, 0.025, 0.05}

Trains ONE balanced model on the full dataset per combo, then evaluates EDM
validation MAE across synthetic test prevalences — same protocol as the binary
new approach (run_edm_binary_new.py) but using multiclass EDM + models.

Usage:
    python tune_edmn_multiclass.py --dataset beans
    python tune_edmn_multiclass.py --dataset beans wine yeast
    python tune_edmn_multiclass.py --all
    python tune_edmn_multiclass.py --all --epochs 40 --patience 10 --warmup 8

Output:
    ../results_ablation/best_params_multiclass.json
    Keys per dataset: {"temperature": T, "bins": b, "epsilon": e, "edm_val_mae": mae}
"""

import os, copy, json, argparse, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_loader, multiclass_models, edm

# ---------------------------------------------------------------------------
# All 22 multiclass datasets
# ---------------------------------------------------------------------------
ALL_DATASETS = [
    'beans', 'conc',  'drugs', 'yeast', 'gest',  'hars',  'micro',
    'wine',  'turk',  'bike',  'blog',  'diam',  'ener',  'fifa',
    'gasd',  'news',  'nurse', 'optd',  'pend',  'rice',  'vgame',
    'thrm',
]

# Search grid
TEMP_CANDIDATES    = [1.5, 2.0, 2.5, 3.0]
BINS_CANDIDATES    = [30, 60, 90]
EPSILON_CANDIDATES = [0.0, 0.01, 0.025, 0.05]

# ---------------------------------------------------------------------------
# Prevalence tables (from run_ablation.py)
# ---------------------------------------------------------------------------
test_2_prevalences  = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5],
                       [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]]
test_3_prevalences  = [[0.1, 0.7, 0.2], [0.55, 0.1, 0.35], [0.35, 0.55, 0.1],
                       [0.4, 0.25, 0.35], [0.0, 0.05, 0.95]]
test_4_prevalences  = [[0.65, 0.25, 0.05, 0.05], [0.2, 0.25, 0.3, 0.25],
                       [0.45, 0.15, 0.2, 0.2], [0.2, 0.0, 0.0, 0.8], [0.3, 0.25, 0.35, 0.1]]
test_5_prevalences  = [[0.15, 0.1, 0.65, 0.1, 0], [0.45, 0.1, 0.3, 0.05, 0.1],
                       [0.2, 0.25, 0.25, 0.1, 0.2], [0.35, 0.05, 0.05, 0.05, 0.5],
                       [0.05, 0.25, 0.15, 0.15, 0.4]]
test_6_prevalences  = [[0.15, 0.1, 0.55, 0.1, 0.0, 0.1], [0.4, 0.1, 0.25, 0.05, 0.1, 0.1],
                       [0.2, 0.2, 0.2, 0.1, 0.2, 0.1], [0.35, 0.05, 0.05, 0.05, 0.05, 0.45],
                       [0.05, 0.25, 0.15, 0.15, 0.1, 0.3]]
test_7_prevalences  = [[0.1, 0.1, 0.1, 0.5, 0.1, 0.0, 0.1], [0.4, 0.1, 0.2, 0.05, 0.1, 0.1, 0.05],
                       [0.15, 0.2, 0.15, 0.1, 0.2, 0.1, 0.1], [0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.45],
                       [0.05, 0.25, 0.1, 0.15, 0.1, 0.3, 0.05]]
test_10_prevalences = [[0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0, 0.1, 0.05, 0.05],
                       [0.2, 0.05, 0.15, 0.05, 0.1, 0.15, 0.05, 0.05, 0.1, 0.1],
                       [0, 0.1, 0.05, 0.1, 0.05, 0.1, 0.1, 0.15, 0.15, 0.2],
                       [0.05, 0.05, 0.05, 0.35, 0.15, 0.05, 0, 0.1, 0.1, 0.1],
                       [0.05, 0.1, 0.1, 0.15, 0.1, 0.15, 0.05, 0.1, 0.1, 0.1]]

test_prev_dictionary = {
    2: test_2_prevalences,  3: test_3_prevalences,
    4: test_4_prevalences,  5: test_5_prevalences,
    6: test_6_prevalences,  7: test_7_prevalences,
    10: test_10_prevalences,
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Tune EDMN multiclass hyperparameters')
parser.add_argument('--dataset',    type=str, nargs='+', default=[])
parser.add_argument('--all',        action='store_true', help='Tune all 22 multiclass datasets')
parser.add_argument('--output',     type=str,
                    default='../results_ablation/best_params_multiclass.json')
parser.add_argument('--epochs',     type=int,   default=80)
parser.add_argument('--patience',   type=int,   default=15)
parser.add_argument('--warmup',     type=int,   default=15)
parser.add_argument('--lr',         type=float, default=0.001)
parser.add_argument('--batch_size', type=int,   default=64)
parser.add_argument('--dm',         type=str,   default='HD')
parser.add_argument('--seed',       type=int,   default=42)
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
# Collect predictions: returns (y_true, y_pred, max_probs)
# multiclass model returns (logits, softmax_probs) — we take argmax + max prob
# ---------------------------------------------------------------------------
def collect_preds(model, loader, temperature):
    model.eval()
    ys, preds, max_probs = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            logits, probs = model(Xb.to(device), temperature)
            mp, pred = torch.max(probs, dim=1)
            max_probs.extend(mp.cpu().numpy().tolist())
            preds.extend(pred.cpu().numpy().tolist())
            ys.extend(yb.numpy().tolist())
    return np.array(ys), np.array(preds), np.array(max_probs)


# ---------------------------------------------------------------------------
# EDM validation MAE over synthetic test prevalences
# ---------------------------------------------------------------------------
def edm_val_mae(model, X, y, num_classes, temperature, bins, dm_name):
    if num_classes not in test_prev_dictionary:
        return float('inf')

    classes, counts = np.unique(y, return_counts=True)
    minority = counts.min()
    ctx_per  = max(15, int(0.20 * minority))
    ctx_per  = min(ctx_per, minority // 2)

    # Reserve balanced context from each class
    ctx_idx = []
    for c in classes:
        idx = np.where(y == c)[0].copy()
        np.random.shuffle(idx)
        ctx_idx.extend(idx[:ctx_per].tolist())
    ctx_idx = np.array(ctx_idx)

    ctx_loader = DataLoader(
        TensorDataset(torch.tensor(X[ctx_idx], dtype=torch.float32),
                      torch.tensor(y[ctx_idx], dtype=torch.long)),
        batch_size=128)
    y_ctx, y_ctx_pred, probs_ctx = collect_preds(model, ctx_loader, temperature)

    # Pre-compute neighbourhood steps
    nb_steps = np.array(list(edm.get_steps(num_classes)))
    for i in range(num_classes):
        nb_steps = np.vstack([nb_steps, edm.get_additional_neighbors(num_classes, step=(i + 2))])

    norm_cm = edm.get_coeficient_matrix(y_ctx, y_ctx_pred, num_classes)

    total_mae, n_valid = 0.0, 0

    for prev in test_prev_dictionary[num_classes]:
        prev = np.array(prev)
        if prev.sum() == 0:
            continue

        # Compute feasible test size
        n_test = minority * 2
        for c_idx, c in enumerate(classes):
            if prev[c_idx] > 0:
                n_test = min(n_test, int(counts[c_idx] * 0.3 / prev[c_idx]))
        n_test = max(n_test, 20)

        te_counts = np.round(n_test * prev).astype(int)
        if np.any(te_counts == 0) and np.any(prev > 0):
            continue

        # Sample test set disjoint from context
        try:
            te_idx_parts = []
            for c_idx, c in enumerate(classes):
                pool = np.setdiff1d(np.where(y == c)[0], ctx_idx)
                if len(pool) < te_counts[c_idx]:
                    raise ValueError('not enough samples')
                chosen = np.random.choice(pool, te_counts[c_idx], replace=False)
                te_idx_parts.append(chosen)
            te_idx = np.concatenate(te_idx_parts)
        except ValueError:
            continue

        te_loader = DataLoader(
            TensorDataset(torch.tensor(X[te_idx], dtype=torch.float32),
                          torch.tensor(y[te_idx], dtype=torch.long)),
            batch_size=128)
        y_te, y_te_pred, probs_te = collect_preds(model, te_loader, temperature)

        try:
            tr_hist, _ = edm.get_train_distributions(y_ctx, y_ctx_pred, probs_ctx, num_classes, bins)
            te_hist, _ = edm.get_test_distributions(y_te_pred, probs_te, num_classes, bins)
            ini_sol, ini_sol_pred = edm.get_init_solution(norm_cm, y_te_pred, num_classes)

            best_est, best_dist = None, float('inf')
            for sol in [ini_sol, ini_sol_pred]:
                est, dist = edm.get_estimation(tr_hist, te_hist, num_classes, nb_steps, sol, dm_name)
                if dist < best_dist:
                    best_dist, best_est = dist, est

            act = edm.get_actual_count(y_te, num_classes)
            est_counts = np.sum(best_est, axis=0)
            total_mae += edm.AE(act, est_counts, num_classes)
            n_valid   += 1
        except Exception:
            continue

    return total_mae / n_valid if n_valid > 0 else float('inf')


# ---------------------------------------------------------------------------
# Train one model with (T, bins, epsilon), return best EDM val MAE
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

    model     = multiclass_models.getModel(dataset_name, input_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=pos_w, label_smoothing=epsilon)

    best_mae   = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits, _ = model(Xb, temperature)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch >= args.warmup:
            mae = edm_val_mae(model, X, y, num_classes, temperature, bins, args.dm)
            if mae < best_mae:
                best_mae   = mae
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    break

    model.load_state_dict(best_state)
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
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
    except Exception as e:
        print(f'  ERROR loading data: {e}\n')
        continue

    # Remap labels to 0..L-1
    classes = np.unique(y)
    if not np.array_equal(classes, np.arange(len(classes))):
        mapping = {c: i for i, c in enumerate(classes)}
        y = np.array([mapping[yi] for yi in y])

    num_classes = len(np.unique(y))
    print(f'  N={len(X)}  D={X.shape[1]}  classes={num_classes}')

    if num_classes < 3:
        print(f'  SKIP — use tune_edmn_binary.py for binary datasets\n')
        continue

    if num_classes not in test_prev_dictionary:
        print(f'  SKIP — no prevalence table for {num_classes} classes\n')
        continue

    results = {}
    best_mae_so_far = float('inf')

    for i, (T, b, e) in enumerate(grid, 1):
        print(f'  [{i:>2}/{len(grid)}] T={T}  bins={b:>2}  eps={e:.3f} ... ', end='', flush=True)
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
