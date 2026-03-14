"""
pipeline_efa_with_pupil.py  --  Same best EFA config + pupil_reward_peak
=========================================================================
Uses the exact same best config found by autoresearch:
  6 core features + pupil_reward_peak (7 total), 3 delta pairs,
  quartimax rotation, 2 factors, RobustScaler
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import FactorAnalysis

sys.path.insert(0, '.')
from prepare_efa import (load_day_level, evaluate_stability,
                          evaluate_index_quality)

PUPIL_PEAK_CSV = Path(__file__).parent / "output" / "pupil_reward_peak.csv"


def load_mouse_level_data_with_pupil(features, delta_pairs):
    """Same as prepare_efa.load_mouse_level_data but merges pupil_reward_peak first."""
    df = load_day_level()

    pupil = pd.read_csv(PUPIL_PEAK_CSV)
    pupil['mouse_key'] = pupil['mouse_key'].astype(str)
    pupil['day_index'] = pupil['day_index'].astype(int)
    df = df.merge(pupil[['mouse_key', 'day_index', 'pupil_reward_peak']],
                  on=['mouse_key', 'day_index'], how='left')

    features = [f for f in features if f in df.columns]

    PERIOD_ORDER = ['Pre', 'During', 'Post', 'Withdrawal', 'Re-exposure']
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

    X_delta = np.array(all_deltas).T
    groups_arr = np.array([
        mouse_period[mouse_period['mouse_key'] == m]['Group'].values[0]
        for m in mice
    ])

    return X_delta, np.array(mice), groups_arr, col_names


def run():
    features = ['RequirementLast', 'lick_freq_per_min', 'bout_n',
                'rew_n', 'rew_freq_per_min',
                'Requirement_speed_per_min',
                'pupil_reward_peak']
    delta_pairs = [('Post', 'Pre'), ('Re-exposure', 'Pre'), ('Withdrawal', 'Pre')]

    X_raw, mice, groups, col_names = load_mouse_level_data_with_pupil(
        features=features, delta_pairs=delta_pairs
    )

    nan_frac = np.mean(np.isnan(X_raw), axis=0)
    keep_cols = nan_frac < 0.50
    X_raw = X_raw[:, keep_cols]
    col_names = [c for c, k in zip(col_names, keep_cols) if k]

    X = X_raw.copy()
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            med = np.nanmedian(X[:, j])
            X[mask, j] = med if np.isfinite(med) else 0.0

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    n_factors = 2
    fa = FactorAnalysis(n_components=n_factors, rotation='quartimax', random_state=42)
    scores = fa.fit_transform(X_scaled)
    loadings = fa.components_.T

    total_var = np.sum(np.var(X_scaled, axis=0))
    explained_var = np.sum(fa.components_ ** 2, axis=1)
    var_explained = float(np.sum(explained_var) / total_var) if total_var > 0 else 0.0

    def loadings_func(X_sub):
        f = FactorAnalysis(n_components=n_factors, rotation='quartimax', random_state=42)
        f.fit(X_sub)
        return f.components_.T

    stability, stability_std, all_corrs = evaluate_stability(loadings_func, X_scaled, n_splits=200)

    n_high_loading = int(np.sum(np.any(np.abs(loadings) > 0.4, axis=1)))
    quality = evaluate_index_quality(stability, var_explained, n_high_loading, len(col_names))

    from scipy.stats import mannwhitneyu
    active_mask = groups == 'Active'
    group_sep_p = []
    for f_idx in range(n_factors):
        try:
            _, p = mannwhitneyu(scores[active_mask, f_idx], scores[~active_mask, f_idx])
            group_sep_p.append(p)
        except Exception:
            group_sep_p.append(1.0)

    print(f"METRIC stability={stability:.6f}")
    print(f"METRIC var_explained={var_explained:.6f}")
    print(f"METRIC n_high_loading={n_high_loading}")
    print(f"METRIC quality_score={quality:.6f}")
    print(f"N_FEATURES={len(col_names)}")
    print(f"N_FACTORS={n_factors}")
    print(f"METHOD=FactorAnalysis_quartimax")
    print(f"SCALING=RobustScaler")

    expl_per_factor = explained_var / total_var if total_var > 0 else np.zeros(n_factors)
    for f_idx in range(n_factors):
        top_idx = np.argsort(np.abs(loadings[:, f_idx]))[::-1][:5]
        top_feats = [(col_names[i], loadings[i, f_idx]) for i in top_idx]
        print(f"FACTOR_{f_idx+1}_top: {top_feats}")
        print(f"FACTOR_{f_idx+1}_var: {expl_per_factor[f_idx]:.4f}")
        print(f"FACTOR_{f_idx+1}_group_p: {group_sep_p[f_idx]:.6f}")

    return {
        'stability': stability,
        'var_explained': var_explained,
        'quality_score': quality,
        'n_high_loading': n_high_loading,
        'loadings': loadings,
        'scores': scores,
        'mice': mice,
        'groups': groups,
        'col_names': col_names,
        'model': fa,
        'scaler': scaler,
        'X_scaled': X_scaled,
        'all_corrs': all_corrs,
        'group_sep_p': group_sep_p,
    }


if __name__ == "__main__":
    run()
