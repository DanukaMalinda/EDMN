#!/bin/bash
# run_edmn.sh — Run EDMN on a dataset and return MAE.
#
# The dataset must be registered in data_loader.py first.
# Run --setup to see step-by-step instructions for adding a new dataset.
#
# Usage:
#   ./run_edmn.sh --dataset cappl --type binary
#   ./run_edmn.sh --dataset beans --type multiclass
#   ./run_edmn.sh --setup
#
# Optional pass-through flags (same for both types):
#     --temperature <float>   Temperature for training variants 3 & 5 (default: 2.0)
#     --bins <int>            Histogram bins (default: 30)
#     --epochs <int>          Max epochs (default: 50)
#     --dm <HD|SE|MH|...>     Distance metric (default: HD)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Setup instructions
# ---------------------------------------------------------------------------
print_setup() {
cat <<'EOF'

  ============================================================
   HOW TO ADD A NEW DATASET TO EDMN
  ============================================================

  Follow these 4 steps. Use data/credit_appl/ as a reference.

  ------------------------------------------------------------
  STEP 1 — Create a folder under data/
  ------------------------------------------------------------

    mkdir data/<your_dataset>
    touch data/<your_dataset>/__init__.py

  ------------------------------------------------------------
  STEP 2 — Create prep.py (loads and cleans raw data)
  ------------------------------------------------------------

    data/<your_dataset>/prep.py

    Example:
    --------------------------------------------------------
    import pandas as pd
    import os

    def prep_data():
        path = os.path.join(os.path.dirname(__file__), 'data.csv')
        df = pd.read_csv(path)
        # clean, encode categoricals, drop NaNs, etc.
        return df
    --------------------------------------------------------

    The raw data file (CSV, Excel, etc.) should live in the
    same folder: data/<your_dataset>/data.csv

  ------------------------------------------------------------
  STEP 3 — Create <dataset>.py (returns X, y as numpy arrays)
  ------------------------------------------------------------

    data/<your_dataset>/<your_dataset>.py

    Example:
    --------------------------------------------------------
    from data.<your_dataset> import prep
    import numpy as np

    def load_data():
        df = prep.prep_data()
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)
        return X, y
    --------------------------------------------------------

    X  — 2D float array  (n_samples, n_features)
    y  — 1D int array    (n_samples,) with labels 0, 1, 2, ...

  ------------------------------------------------------------
  STEP 4 — Register in src/data_loader.py
  ------------------------------------------------------------

    Open src/data_loader.py and add two things:

    a) Import at the top:
       from data.<your_dataset> import <your_dataset>

    b) Add an elif branch in load_data():
       elif dataset == '<your_dataset>':
           X_n, y_n = <your_dataset>.load_data()

  ------------------------------------------------------------
  STEP 5 — Run EDMN
  ------------------------------------------------------------

    cd src/
    ./run_edmn.sh --dataset <your_dataset> --type binary
    ./run_edmn.sh --dataset <your_dataset> --type multiclass

  ============================================================

EOF
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DATASET=""
TYPE=""
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"; shift 2 ;;
        --type)
            TYPE="$2"; shift 2 ;;
        --setup)
            print_setup; exit 0 ;;
        --help|-h)
            sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *)
            PASSTHROUGH+=("$1"); shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [[ -z "$DATASET" ]]; then
    echo "ERROR: --dataset <name> is required." >&2
    echo "       Run: $0 --setup  to see how to add a new dataset." >&2
    exit 1
fi

if [[ -z "$TYPE" ]]; then
    echo "ERROR: --type binary|multiclass is required." >&2
    exit 1
fi

if [[ "$TYPE" != "binary" && "$TYPE" != "multiclass" ]]; then
    echo "ERROR: --type must be binary or multiclass (got: $TYPE)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo ""
echo "  Dataset : $DATASET"
echo "  Type    : $TYPE"
[[ ${#PASSTHROUGH[@]} -gt 0 ]] && echo "  Args    : ${PASSTHROUGH[*]}"
echo ""

cd "$SCRIPT_DIR"

TMPLOG="$(mktemp)"
trap 'rm -f "$TMPLOG"' EXIT

if [[ "$TYPE" == "binary" ]]; then
    python run_ablation_binary.py \
        --dataset "$DATASET" \
        "${PASSTHROUGH[@]}" \
        2>&1 | tee "$TMPLOG"
else
    python run_ablation.py \
        --dataset "$DATASET" \
        "${PASSTHROUGH[@]}" \
        2>&1 | tee "$TMPLOG"
fi

EXIT_CODE="${PIPESTATUS[0]}"
[[ "$EXIT_CODE" -ne 0 ]] && { echo "ERROR: run failed (exit $EXIT_CODE)" >&2; exit "$EXIT_CODE"; }

# Both scripts print the same summary line:
# "  Full EDMN                           0.0312  ..."  ($1=Full $2=EDMN $3=MAE)
MAE=$(grep "Full EDMN" "$TMPLOG" | awk '{print $3}' | tail -1)

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
if [[ -z "$MAE" ]]; then
    echo "" >&2
    echo "ERROR: could not extract MAE from output." >&2
    exit 1
fi

echo ""
echo "=========================================="
echo "  RESULT"
echo "  Dataset : $DATASET"
echo "  Type    : $TYPE"
echo "  MAE     : $MAE"
echo "=========================================="
