# EDMN
EDMN: Enhanced Distribution Matching with Quantification-Aware Neural Networks for Quantification

Quantification is the task of estimating the class distribution in an independent, unlabeled dataset. Existing
distribution-matching methods rely on classifiers that produce overconfident, low-entropy outputs, causing
inter-class conditional distributions to collapse and rendering class distribution reconstruction unreliable. To
overcome this, EDMN introduces a two-step framework that trains a quantification-aware neural network with
a novel composite loss that enforces sufficient output entropy while preserving class identity, and then applies
enhanced distribution matching over inter-class conditional probability distributions. Experimental results
on 58 benchmark datasets show that the proposed approach outperforms existing quantification methods
achieving the lowest MAE on 79% of the datasets, including 19 of 22 multiclass and 27 of 36 binary datasets

---

## Requirements

```bash
pip install -r requirements.txt
```

All scripts must be run from the `src/` directory:

```bash
cd src/
```

---

## Running EDMN

Both scripts run five ablation variants in sequence for every experimental
configuration (train/test prevalence combination). The final summary line
`Full EDMN` is the main EDMN result.

### Binary datasets

```bash
python run_ablation_binary.py --dataset <name>
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--distance_metric` | `JD` | Distance metric (HD, SE, MH, TS, …) |
| `--temperature` | `1.5` | Softmax temperature for variants 3 & 5 |
| `--bins` | `90` | Histogram bin count |
| `--epochs` | `50` | Training epochs per variant |

Examples:

```bash
python run_ablation_binary.py --dataset cappl
python run_ablation_binary.py --dataset spam_b --temperature 3.0 --bins 60
python run_ablation_binary.py --dataset wine_b --distance_metric SE
```

Output is saved to `../results_ablation_binary/ablation_binary_<dataset>_<dm>.csv`.

### Multiclass datasets

```bash
python run_ablation.py --dataset <name>
```

Same flags as binary. Examples:

```bash
python run_ablation.py --dataset beans
python run_ablation.py --dataset yeast --temperature 2.5 --bins 30
python run_ablation.py --dataset gasd --distance_metric HD --epochs 100
```

Output is saved to `../results_ablation/<dataset>_<dm>.csv`.

### Reading the results

Both scripts print a summary table at the end. The five variants are:

| Variant | What it tests |
|---|---|
| Baseline NN + EDM | Plain CE loss, no temperature, no EDM loss |
| NN + Label Smoothing + EDM | Adds label smoothing (LS=0.025) |
| NN + Temperature Scaling + EDM | Adds softmax temperature during training |
| NN + LDM Loss + EDM | Adds EDM loss in gradient (no LS) |
| **Full EDMN** | **CE + LS + temperature + EDM loss — main result** |

---

## Running the Ablation Study

The ablation study is the same command as running EDMN — the scripts run all
five variants automatically. Use the summary table to compare variants.

### Binary ablation

```bash
python run_ablation_binary.py --dataset cappl
python run_ablation_binary.py --dataset musk --temperature 3.0
```

### Multiclass ablation

```bash
python run_ablation.py --dataset beans
python run_ablation.py --dataset wine --temperature 2.0 --bins 30
```

---

## Hyperparameter Tuning

Grid search over temperature, histogram bins, and label smoothing epsilon.
Trains one balanced model per combination and selects by EDM validation MAE.

Search grid:

| Hyperparameter | Values |
|---|---|
| `temperature` | 1.5, 2.0, 2.5, 3.0 |
| `bins` | 30, 60, 90 |
| `epsilon` (label smoothing) | 0.0, 0.01, 0.025, 0.05 |

48 combinations per dataset.

### Binary tuning

```bash
# Tune a single dataset
python tune_edmn_binary.py --dataset cappl

# Tune multiple datasets
python tune_edmn_binary.py --dataset cappl bc_cont wine_b

# Tune all 36 binary datasets
python tune_edmn_binary.py --all

# Quick scan with fewer epochs
python tune_edmn_binary.py --all --epochs 40 --patience 10 --warmup 8

# Narrow the grid
python tune_edmn_binary.py --all --temps 2.0 3.0 --bins_list 30 60
```

Results saved to `../results_ablation_binary/best_params_binary.json`.

### Multiclass tuning

```bash
# Tune a single dataset
python tune_edmn_multiclass.py --dataset beans

# Tune multiple datasets
python tune_edmn_multiclass.py --dataset beans yeast wine

# Tune all 22 multiclass datasets
python tune_edmn_multiclass.py --all

# Quick scan
python tune_edmn_multiclass.py --all --epochs 40 --patience 10 --warmup 8
```

Results saved to `../results_ablation/best_params_multiclass.json`.

Each entry in the JSON looks like:

```json
"cappl": {
  "temperature": 3.0,
  "bins": 30,
  "epsilon": 0.0,
  "edm_val_mae": 0.031200
}
```

---

## KDEy-ML Comparison

`run_quapy_comparison.py` runs KDEy-ML from the QuaPy library on the same
datasets and with the exact same train/test sampling protocol as EDMN, making
the comparison directly fair.

### Implementation details

**Identical sampling protocol.**
The script uses the same `synthetic_draw()` function as `run_ablation.py`,
producing the same train/test splits across the same grid of prevalence
configurations (`dt_distr × train_distr × test_distr`).

**Leftover calibration.**
The standard KDEy-ML workflow fits the KDE calibration on a held-out split
from the training set, which can leak prevalence bias. Here `synthetic_draw()`
returns a third partition — leftover samples not drawn into train or test —
at the natural dataset prevalence. This leftover set is passed as
`val_split=(X_leftover, y_leftover)` to KDEy-ML, so KDE calibration uses
unbiased, prevalence-neutral samples disjoint from both train and test.

```
full dataset
├── train      — biased prevalence (controlled by train_distr)
├── test       — biased prevalence (controlled by test_distr)
└── leftover   — natural prevalence → used for KDE calibration
```

**KDEy-ML isolation.**
The original comparison script included eight methods (PACC+, DM-T, DM-HD,
DM-CS, EMQ, KDEy-HD, KDEy-CS, KDEy-ML). This version retains only KDEy-ML (best method),
removing all other methods, their imports, and the `--methods` flag. The
`make_kdey_ml()` function replaces the general `make_methods()` factory.

### How to run

```bash
# Single dataset
python run_quapy_comparison.py --dataset beans
python run_quapy_comparison.py --dataset yeast

# All 22 multiclass datasets
python run_quapy_comparison.py --all
```

The script prints per-dataset MAE and, when `--all` is used, a summary table:

```
===================================
  MULTI-DATASET MAE SUMMARY — KDEy-ML
===================================
  Dataset        MAE
  ------------   ----------
  beans          0.04231
  conc           0.09812
  ...
  MEAN           0.07145
===================================
```

### Adding a new dataset

Run `./run_edmn.sh --setup` for step-by-step instructions on creating
`prep.py`, the dataset loader module, and registering it in `data_loader.py`.

---

## Third-party libraries

This project includes a bundled copy of
[QuaPy](https://github.com/HLT-ISTI/QuaPy)
(BSD 3-Clause License, Copyright 2020 HLT-ISTI) in the `QuaPy/` directory,
used solely for the KDEy-ML comparison in `run_quapy_comparison.py`.
