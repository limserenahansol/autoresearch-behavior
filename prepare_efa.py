"""
prepare_efa.py  --  Data Loader & Evaluation for Addiction Index (EFA / PCA)
============================================================================
DO NOT MODIFY.  This file is the fixed ground truth for addiction index experiments.

Provides:
  - load_mouse_level_data() : aggregates day-level → per-mouse-per-period medians,
                              then computes delta scores (Post-Pre, Reexp-Pre, etc.)
  - evaluate_stability()    : split-half cross-validation of factor loadings
  - evaluate_index_quality(): combined metric (stability + variance explained + interpretability)

Data source:
  K:/addiction_concate_Dec_2025/longitudinal_outputs/run_009/
  figs/modules_5_to_11/stats/features_day_level.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(
    r"K:\addiction_concate_Dec_2025\longitudinal_outputs\run_009"
    r"\figs\modules_5_to_11\stats\features_day_level.csv"
)

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


def load_day_level():
    """Load raw day-level data."""
    df = pd.read_csv(DATA_PATH)
    df['mouse_key'] = df['mouse_key'].astype(str)
    df['Group'] = df['Group'].astype(str)
    df['Period'] = pd.Categorical(df['Period'], categories=PERIOD_ORDER, ordered=True)
    df['day_index'] = df['day_index'].astype(int)
    return df


def load_mouse_level_data(features=None, delta_pairs=None):
    """
    Aggregate day-level data to per-mouse medians, then compute delta scores.

    Args:
        features: list of feature column names to use
        delta_pairs: list of (phase_A, phase_B) tuples → delta = A - B
                     e.g. [('Post', 'Pre'), ('Re-exposure', 'Pre')]

    Returns:
        X_delta     : ndarray [n_mice, n_features * n_deltas]
        mouse_keys  : ndarray [n_mice]
        groups      : ndarray [n_mice] ("Active" / "Passive")
        col_names   : list of str  (feature_deltaName)
    """
    if features is None:
        features = ALL_CANDIDATE_FEATURES
    if delta_pairs is None:
        delta_pairs = [('Post', 'Pre'), ('Re-exposure', 'Pre'), ('Withdrawal', 'Pre')]

    df = load_day_level()
    features = [f for f in features if f in df.columns]

    mouse_period = df.groupby(['mouse_key', 'Group', 'Period'])[features].median().reset_index()

    mice = sorted(mouse_period['mouse_key'].unique())
    all_deltas = []
    col_names = []

    for phase_a, phase_b in delta_pairs:
        for feat in features:
            delta_vals = []
            for m in mice:
                a_rows = mouse_period[(mouse_period['mouse_key'] == m) &
                                      (mouse_period['Period'] == phase_a)]
                b_rows = mouse_period[(mouse_period['mouse_key'] == m) &
                                      (mouse_period['Period'] == phase_b)]
                val_a = a_rows[feat].values[0] if len(a_rows) > 0 else np.nan
                val_b = b_rows[feat].values[0] if len(b_rows) > 0 else np.nan
                delta_vals.append(val_a - val_b)
            all_deltas.append(delta_vals)
            col_names.append(f"{feat}_{phase_a}-{phase_b}")

    X_delta = np.array(all_deltas).T  # [n_mice, n_features * n_deltas]
    groups_arr = np.array([
        mouse_period[mouse_period['mouse_key'] == m]['Group'].values[0]
        for m in mice
    ])

    return X_delta, np.array(mice), groups_arr, col_names


def evaluate_stability(loadings_func, X, n_splits=100, random_state=42):
    """
    Split-half cross-validation of factor stability.

    Args:
        loadings_func: callable(X_subset) → loadings matrix [n_features, n_factors]
        X            : ndarray [n_samples, n_features] (already cleaned/scaled)
        n_splits     : number of random splits
        random_state : seed

    Returns:
        mean_corr    : mean absolute correlation of matched loadings across splits
        std_corr     : std of correlations
        all_corrs    : list of per-split correlations
    """
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    all_corrs = []

    for _ in range(n_splits):
        idx = rng.permutation(n)
        half = n // 2
        X_a, X_b = X[idx[:half]], X[idx[half:2*half]]

        try:
            L_a = loadings_func(X_a)
            L_b = loadings_func(X_b)
        except Exception:
            continue

        if L_a.shape != L_b.shape or L_a.shape[1] == 0:
            continue

        n_factors = L_a.shape[1]
        factor_corrs = []
        for f in range(n_factors):
            r, _ = spearmanr(L_a[:, f], L_b[:, f])
            factor_corrs.append(abs(r) if np.isfinite(r) else 0.0)
        all_corrs.append(np.mean(factor_corrs))

    if len(all_corrs) == 0:
        return 0.0, 0.0, []

    return float(np.mean(all_corrs)), float(np.std(all_corrs)), all_corrs


def evaluate_index_quality(stability, var_explained, n_high_loading, n_features):
    """
    Combined quality metric for the addiction index.

    Args:
        stability      : split-half correlation (0-1)
        var_explained  : fraction of total variance explained by retained factors (0-1)
        n_high_loading : number of features with |loading| > 0.4 on at least one factor
        n_features     : total number of features used

    Returns:
        quality_score  : weighted combination (higher = better)
    """
    interpretability = n_high_loading / max(1, n_features)
    quality = 0.50 * stability + 0.30 * var_explained + 0.20 * interpretability
    return float(quality)


if __name__ == "__main__":
    X, mice, groups, cols = load_mouse_level_data()
    print(f"Delta matrix: {X.shape[0]} mice × {X.shape[1]} delta-features")
    print(f"Groups: Active={np.sum(groups=='Active')}, Passive={np.sum(groups=='Passive')}")
    print(f"Feature columns (first 10): {cols[:10]}")
    nan_pct = np.mean(np.isnan(X)) * 100
    print(f"NaN %: {nan_pct:.1f}%")
