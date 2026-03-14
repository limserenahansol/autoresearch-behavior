"""
generate_addiction_trajectory.py  --  Addiction Score Trajectory Across Days
============================================================================
Projects each mouse's daily behavioral data onto the EFA factor loadings
to compute a daily addiction score, then plots spaghetti + mean trajectory
separately for Active and Passive groups (matching the MATLAB style).

Output:
  output/figures_efa/10_addiction_trajectory.png
  output/figures_efa/11_addiction_trajectory_combined.png
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import FactorAnalysis

sys.path.insert(0, '.')
from prepare_efa import load_day_level

OUTPUT = Path(__file__).parent / "output" / "figures_efa"
OUTPUT.mkdir(parents=True, exist_ok=True)

COL_ACTIVE = '#E53935'
COL_PASSIVE = '#1E88E5'

FEATURES = ['RequirementLast', 'lick_freq_per_min', 'bout_n',
            'rew_n', 'rew_freq_per_min', 'Requirement_speed_per_min']

PERIOD_DAYS = {
    'Pre': (3, 5),
    'During': (6, 10),
    'Post': (11, 13),
    'Withdrawal': (14, 16),
    'Re-exposure': (17, 18),
}
PERIOD_COLORS = {
    'Pre': '#BDBDBD',
    'During': '#42A5F5',
    'Post': '#66BB6A',
    'Withdrawal': '#FFA726',
    'Re-exposure': '#FF7043',
}


def compute_daily_scores():
    """
    For each mouse-day, compute addiction score by:
    1. Get the mouse's Pre-phase median for each feature
    2. Compute delta = day_value - Pre_median
    3. Scale using the same RobustScaler fitted on per-mouse deltas
    4. Project onto the EFA loadings to get factor scores
    5. Combine into composite addiction score
    """
    df = load_day_level()
    feats = [f for f in FEATURES if f in df.columns]

    mice = sorted(df['mouse_key'].unique())
    groups_map = dict(zip(df['mouse_key'], df['Group']))

    pre_medians = {}
    for m in mice:
        pre_rows = df[(df['mouse_key'] == m) & (df['Period'] == 'Pre')]
        pre_medians[m] = pre_rows[feats].median()

    # First pass: compute all per-mouse-per-period medians for EFA fitting
    from prepare_efa import load_mouse_level_data
    delta_pairs = [('Post', 'Pre'), ('Re-exposure', 'Pre'), ('Withdrawal', 'Pre')]
    X_mouse, mice_arr, groups_arr, col_names = load_mouse_level_data(
        features=FEATURES, delta_pairs=delta_pairs
    )
    nan_frac = np.mean(np.isnan(X_mouse), axis=0)
    keep = nan_frac < 0.50
    X_mouse = X_mouse[:, keep]
    col_names = [c for c, k in zip(col_names, keep) if k]

    X_clean = X_mouse.copy()
    for j in range(X_clean.shape[1]):
        mask = np.isnan(X_clean[:, j])
        if mask.any():
            med = np.nanmedian(X_clean[:, j])
            X_clean[mask, j] = med if np.isfinite(med) else 0.0

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_clean)

    fa = FactorAnalysis(n_components=2, rotation='quartimax', random_state=42)
    fa.fit(X_scaled)
    loadings = fa.components_.T  # [n_features, 2]

    var_per_factor = np.var(fa.transform(X_scaled), axis=0)
    weights = var_per_factor / var_per_factor.sum()

    # Now compute daily scores
    # For each day, create a "pseudo-delta" vector relative to that mouse's Pre median
    # We need to map the 6 features to the 18 delta-feature positions
    # The loading structure: features 0-5 = Post-Pre, 6-11 = Reexp-Pre, 12-17 = Withdr-Pre
    # For a daily score, we use the same delta (day - Pre) projected onto all 3 delta slots

    results = []
    for _, row in df.iterrows():
        m = row['mouse_key']
        day = row['day_index']
        group = row['Group']

        pre_med = pre_medians[m]
        day_vals = row[feats].values.astype(float)
        delta = day_vals - pre_med.values

        # Build the 18-feature delta vector (repeat delta for all 3 delta pairs)
        delta_18 = np.tile(delta, 3)

        # Handle NaN
        nan_mask = np.isnan(delta_18)
        delta_18[nan_mask] = 0.0

        # Only keep columns that survived the NaN filter
        delta_18 = delta_18[keep]

        # Scale using the same scaler
        delta_scaled = scaler.transform(delta_18.reshape(1, -1))[0]

        # Project onto loadings
        f1_score = np.dot(delta_scaled, loadings[:, 0])
        f2_score = np.dot(delta_scaled, loadings[:, 1])
        composite = weights[0] * f1_score + weights[1] * f2_score

        results.append({
            'mouse': m,
            'group': group,
            'day': day,
            'factor1': f1_score,
            'factor2': f2_score,
            'composite': composite,
        })

    return pd.DataFrame(results)


def plot_trajectory_separate(df_scores):
    """Two-panel figure: Passive (left, blue) and Active (right, red)"""
    fig, (ax_p, ax_a) = plt.subplots(1, 2, figsize=(22, 9), sharey=True)

    days = sorted(df_scores['day'].unique())

    for ax, group, color, title in [
        (ax_p, 'Passive', COL_PASSIVE, 'Passive Group'),
        (ax_a, 'Active', COL_ACTIVE, 'Active Group'),
    ]:
        grp_data = df_scores[df_scores['group'] == group]
        mice = sorted(grp_data['mouse'].unique())

        # Individual trajectories (thin lines)
        for m in mice:
            mdata = grp_data[grp_data['mouse'] == m].sort_values('day')
            ax.plot(mdata['day'], mdata['composite'], '-o',
                    color=color, alpha=0.35, linewidth=1, markersize=4)

        # Group mean (thick line)
        mean_by_day = grp_data.groupby('day')['composite'].mean()
        sem_by_day = grp_data.groupby('day')['composite'].sem()
        ax.plot(mean_by_day.index, mean_by_day.values, '-o',
                color=color, linewidth=3.5, markersize=8, zorder=5,
                markeredgecolor='black', markeredgewidth=0.8)
        ax.fill_between(mean_by_day.index,
                        mean_by_day.values - sem_by_day.values,
                        mean_by_day.values + sem_by_day.values,
                        color=color, alpha=0.15)

        # Period shading
        for period, (d_start, d_end) in PERIOD_DAYS.items():
            ax.axvspan(d_start - 0.4, d_end + 0.4, alpha=0.06,
                       color=PERIOD_COLORS[period])
            ax.text((d_start + d_end) / 2, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -3.2,
                    period, ha='center', fontsize=9, fontstyle='italic',
                    color=PERIOD_COLORS[period], fontweight='bold')

        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Day', fontsize=14, fontweight='bold')
        ax.set_title(f'Addiction Score Trajectory ({title})',
                     fontsize=16, fontweight='bold', color=color, pad=12)
        ax.set_xticks(days)
        ax.grid(axis='y', alpha=0.15)
        ax.tick_params(labelsize=12)

        n_mice = len(mice)
        ax.text(0.02, 0.98, f'n = {n_mice} mice\nthin = individual\nthick = mean +/- SEM',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    ax_p.set_ylabel('Addiction Score (z)', fontsize=14, fontweight='bold')

    # Add period labels at top
    for period, (d_start, d_end) in PERIOD_DAYS.items():
        mid = (d_start + d_end) / 2
        # Normalize to figure coordinates
        for ax in [ax_p, ax_a]:
            ax.annotate('', xy=(d_end + 0.4, ax.get_ylim()[1] * 0.95),
                        xytext=(d_start - 0.4, ax.get_ylim()[1] * 0.95),
                        arrowprops=dict(arrowstyle='|-|', color=PERIOD_COLORS[period],
                                        linewidth=2))

    fig.suptitle('Addiction Score Trajectory Across Days',
                 fontsize=20, fontweight='bold', y=1.01, color='#0D47A1')

    plt.tight_layout()
    fig.savefig(OUTPUT / '10_addiction_trajectory.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 10_addiction_trajectory.png")


def plot_trajectory_combined(df_scores):
    """Single panel with both groups overlaid"""
    fig, ax = plt.subplots(figsize=(16, 8))

    days = sorted(df_scores['day'].unique())

    for group, color, label in [
        ('Active', COL_ACTIVE, 'Active'),
        ('Passive', COL_PASSIVE, 'Passive'),
    ]:
        grp_data = df_scores[df_scores['group'] == group]
        mice = sorted(grp_data['mouse'].unique())

        for m in mice:
            mdata = grp_data[grp_data['mouse'] == m].sort_values('day')
            ax.plot(mdata['day'], mdata['composite'], '-',
                    color=color, alpha=0.2, linewidth=0.8)

        mean_by_day = grp_data.groupby('day')['composite'].mean()
        sem_by_day = grp_data.groupby('day')['composite'].sem()
        ax.plot(mean_by_day.index, mean_by_day.values, '-o',
                color=color, linewidth=3, markersize=7, label=f'{label} (n={len(mice)})',
                markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        ax.fill_between(mean_by_day.index,
                        mean_by_day.values - sem_by_day.values,
                        mean_by_day.values + sem_by_day.values,
                        color=color, alpha=0.15)

    # Period shading
    for period, (d_start, d_end) in PERIOD_DAYS.items():
        ax.axvspan(d_start - 0.4, d_end + 0.4, alpha=0.07,
                   color=PERIOD_COLORS[period])
        y_top = ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 2
        ax.text((d_start + d_end) / 2, y_top * 0.92, period,
                ha='center', fontsize=11, fontweight='bold',
                color=PERIOD_COLORS[period])

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Day', fontsize=14, fontweight='bold')
    ax.set_ylabel('Addiction Score (z)', fontsize=14, fontweight='bold')
    ax.set_title('Addiction Score Trajectory: Active vs Passive',
                 fontsize=18, fontweight='bold', pad=15, color='#0D47A1')
    ax.set_xticks(days)
    ax.legend(fontsize=13, loc='upper left')
    ax.grid(axis='y', alpha=0.15)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(OUTPUT / '11_addiction_trajectory_combined.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 11_addiction_trajectory_combined.png")


def main():
    print("Computing daily addiction scores...")
    df_scores = compute_daily_scores()

    print(f"  {len(df_scores)} mouse-day observations")
    print(f"  Days: {sorted(df_scores['day'].unique())}")
    print(f"  Active mice: {df_scores[df_scores['group']=='Active']['mouse'].nunique()}")
    print(f"  Passive mice: {df_scores[df_scores['group']=='Passive']['mouse'].nunique()}")

    print("\n[1] Separate panels (Passive | Active)...")
    plot_trajectory_separate(df_scores)

    print("\n[2] Combined overlay...")
    plot_trajectory_combined(df_scores)

    # Save data
    df_scores.to_csv(OUTPUT.parent / 'addiction_scores_daily.csv', index=False)
    print(f"\n  Saved: addiction_scores_daily.csv")
    print(f"\nAll figures in: {OUTPUT}")


if __name__ == "__main__":
    main()
