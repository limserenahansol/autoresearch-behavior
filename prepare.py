"""
prepare.py  --  Data Loader & Evaluation Functions
===================================================
DO NOT MODIFY.  This file is the fixed "ground truth" for all experiments.

Loads the mouse behavior CSV (219 rows x 62 columns) and provides:
  - load_data()              : returns full DataFrame
  - get_feature_matrix()     : extracts numeric X matrix + metadata
  - evaluate_classification(): accuracy, per-mouse accuracy, macro F1
  - evaluate_regression()    : Spearman r, MSE

Data source:
  K:/addiction_concate_Dec_2025/longitudinal_outputs/run_009/
  figs/modules_5_to_11/stats/features_day_level.csv

Dataset:
  14 mice (6 Active, 8 Passive) x 16 days (day 3-18) = 219 rows
  5 periods: Pre(3-5), During(6-10), Post(11-13), Withdrawal(14-16), Re-exposure(17-18)
  Passive mice have NaN for lick/reward metrics during "During" (by design).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

# ── Data path ───────────────────────────────────────────────────────
DATA_PATH = Path(
    r"K:\addiction_concate_Dec_2025\longitudinal_outputs\run_009"
    r"\figs\modules_5_to_11\stats\features_day_level.csv"
)

# ── Feature groups (all columns available in the CSV) ───────────────
BEHAVIORAL_FEATURES = [
    'RequirementLast', 'lick_freq_per_min', 'lick_meanDur_s',
    'lick_medianIEI_s', 'lick_totalDur_s',
    'bout_n', 'bout_meanDur_s', 'bout_totalDur_s',
    'rew_n', 'rew_freq_per_min', 'rew_totalDur_s', 'rew_medianIRI_s',
    'Requirement_cum', 'Requirement_speed_per_day', 'Requirement_speed_per_min',
]
PUPIL_FEATURES = ['pupil_mean']
PHARMA_FEATURES = [
    'Immersion_Latency_s',
    'TST_Pct_Non_moving', 'TST_Pct_Licking', 'TST_Pct_Rearing',
    'TST_Pct_Flinching', 'TST_Pct_HindlimbLicking', 'TST_Pct_Jump',
    'HOT_Pct_Non_moving', 'HOT_Pct_Licking', 'HOT_Pct_Rearing',
    'HOT_Pct_Flinching', 'HOT_Pct_HindlimbLicking', 'HOT_Pct_Jump',
]
ALL_CANDIDATE_FEATURES = BEHAVIORAL_FEATURES + PUPIL_FEATURES + PHARMA_FEATURES

PERIOD_ORDER = ['Pre', 'During', 'Post', 'Withdrawal', 'Re-exposure']


# ── Data loading ────────────────────────────────────────────────────

def load_data():
    """Load the features CSV and return a clean DataFrame."""
    df = pd.read_csv(DATA_PATH)
    df['mouse_key'] = df['mouse_key'].astype(str)
    df['Group'] = df['Group'].astype(str)
    df['Period'] = pd.Categorical(df['Period'], categories=PERIOD_ORDER, ordered=True)
    df['day_index'] = df['day_index'].astype(int)
    return df


def get_feature_matrix(df, feature_cols=None):
    """Extract numeric feature matrix and metadata arrays.

    Args:
        df: DataFrame from load_data()
        feature_cols: list of column names to use (default: ALL_CANDIDATE_FEATURES)

    Returns:
        X            : ndarray [N x F]
        mouse_keys   : ndarray [N] of str
        groups       : ndarray [N] of str ("Active" / "Passive")
        periods      : ndarray [N] of str
        days         : ndarray [N] of int
        feature_names: list of str
    """
    if feature_cols is None:
        feature_cols = [c for c in ALL_CANDIDATE_FEATURES if c in df.columns]
    else:
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values.astype(float)
    return (
        X,
        df['mouse_key'].values,
        df['Group'].values,
        df['Period'].values.astype(str),
        df['day_index'].values,
        feature_cols,
    )


# ── Evaluation functions ────────────────────────────────────────────

def evaluate_classification(y_true, y_pred, mouse_keys):
    """Compute classification metrics with leave-one-mouse-out awareness.

    Returns dict:
        accuracy      : overall fraction correct
        per_mouse_acc : mean of per-mouse accuracies (robust to imbalance)
        f1_macro      : macro-averaged F1 score
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    accuracy = np.mean(y_true == y_pred)

    mice = np.unique(mouse_keys)
    per_mouse_acc = np.mean([
        np.mean(y_true[mouse_keys == m] == y_pred[mouse_keys == m])
        for m in mice if np.any(mouse_keys == m)
    ])

    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1s.append(2 * prec * rec / max(1e-12, prec + rec) if prec + rec > 0 else 0.0)

    return {'accuracy': accuracy, 'per_mouse_acc': per_mouse_acc, 'f1_macro': np.mean(f1s)}


def evaluate_regression(y_true, y_pred, mouse_keys):
    """Compute regression metrics (e.g. for morphine effect prediction).

    Returns dict: spearman_r, spearman_p, mse
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 4:
        return {'spearman_r': 0.0, 'spearman_p': 1.0, 'mse': 999.0}
    r, p = spearmanr(y_true[ok], y_pred[ok])
    mse = float(np.mean((y_true[ok] - y_pred[ok]) ** 2))
    return {'spearman_r': float(r), 'spearman_p': float(p), 'mse': mse}


# ── Quick sanity check ──────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows, {df['mouse_key'].nunique()} mice")
    print(f"Groups:  {df.groupby('Group')['mouse_key'].nunique().to_dict()}")
    print(f"Periods: {list(df['Period'].cat.categories)}")
    print(f"Available features: {len([c for c in ALL_CANDIDATE_FEATURES if c in df.columns])}")
