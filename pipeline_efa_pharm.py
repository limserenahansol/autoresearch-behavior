"""
pipeline_efa_pharm.py  --  Addiction Index with Pharmacological Data
====================================================================
Same best method as pipeline_efa.py (quartimax, RobustScaler, 2 factors)
but with pharmacological variables added.
"""
import sys
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import FactorAnalysis

sys.path.insert(0, '.')
from prepare_efa import (load_mouse_level_data, evaluate_stability,
                          evaluate_index_quality, PHARMA_FEATURES)


def run():
    # ── 1. Feature selection: core behavioral + pharma ───────────────
    features = ['RequirementLast', 'lick_freq_per_min', 'bout_n',
                'rew_n', 'rew_freq_per_min', 'Requirement_speed_per_min',
                'pupil_mean',
                'Immersion_Latency_s',
                'TST_Pct_Non_moving', 'TST_Pct_Licking', 'TST_Pct_Rearing',
                'HOT_Pct_Non_moving', 'HOT_Pct_Licking', 'HOT_Pct_Rearing']
    delta_pairs = [('Post', 'Pre'), ('Re-exposure', 'Pre'), ('Withdrawal', 'Pre')]

    X_raw, mice, groups, col_names = load_mouse_level_data(
        features=features, delta_pairs=delta_pairs
    )

    # ── 2. Remove columns with >50% NaN ─────────────────────────────
    nan_frac = np.mean(np.isnan(X_raw), axis=0)
    keep_cols = nan_frac < 0.50
    X_raw = X_raw[:, keep_cols]
    col_names = [c for c, k in zip(col_names, keep_cols) if k]

    print(f"Features after NaN filter: {len(col_names)} (from {sum(~keep_cols)} removed)")
    print(f"Removed: {[c for c, k in zip(col_names, keep_cols) if not k][:5]}...")

    # ── 3. Impute remaining NaN with column median ───────────────────
    X = X_raw.copy()
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            med = np.nanmedian(X[:, j])
            X[mask, j] = med if np.isfinite(med) else 0.0

    # ── 4. Scaling ───────────────────────────────────────────────────
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 5. Factor analysis with quartimax rotation ───────────────────
    n_factors = 3
    fa = FactorAnalysis(n_components=n_factors, rotation='quartimax', random_state=42)
    scores = fa.fit_transform(X_scaled)
    loadings = fa.components_.T

    total_var = np.sum(np.var(X_scaled, axis=0))
    explained_var = np.sum(fa.components_ ** 2, axis=1)
    var_explained = float(np.sum(explained_var) / total_var) if total_var > 0 else 0.0

    # ── 6. Evaluate stability ────────────────────────────────────────
    def loadings_func(X_sub):
        f = FactorAnalysis(n_components=n_factors, rotation='quartimax', random_state=42)
        f.fit(X_sub)
        return f.components_.T

    stability, stability_std, _ = evaluate_stability(loadings_func, X_scaled, n_splits=200)

    # ── 7. Interpretability ──────────────────────────────────────────
    n_high_loading = int(np.sum(np.any(np.abs(loadings) > 0.4, axis=1)))

    # ── 8. Quality score ─────────────────────────────────────────────
    quality = evaluate_index_quality(stability, var_explained, n_high_loading, len(col_names))

    # ── 9. Group separation ──────────────────────────────────────────
    from scipy.stats import mannwhitneyu
    active_mask = groups == 'Active'
    group_sep_p = []
    for f_idx in range(n_factors):
        try:
            _, p = mannwhitneyu(scores[active_mask, f_idx], scores[~active_mask, f_idx])
            group_sep_p.append(p)
        except Exception:
            group_sep_p.append(1.0)

    # ── Print results ────────────────────────────────────────────────
    print(f"\nMETRIC stability={stability:.6f}")
    print(f"METRIC var_explained={var_explained:.6f}")
    print(f"METRIC n_high_loading={n_high_loading}")
    print(f"METRIC quality_score={quality:.6f}")
    print(f"N_FEATURES={len(col_names)}")
    print(f"N_FACTORS={n_factors}")
    print(f"METHOD=FactorAnalysis_quartimax_with_pharma")

    expl_per_factor = explained_var / total_var if total_var > 0 else np.zeros(n_factors)
    for f_idx in range(n_factors):
        top_idx = np.argsort(np.abs(loadings[:, f_idx]))[::-1][:5]
        top_feats = [(col_names[i], f'{loadings[i, f_idx]:.3f}') for i in top_idx]
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
        'X_scaled': X_scaled,
        'group_sep_p': group_sep_p,
    }


if __name__ == "__main__":
    run()
