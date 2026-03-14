"""
visualize_efa.py  --  Addiction Index (EFA) Visualization
=========================================================
Generates:
  output/figures_efa/01_factor_loadings.png      - Heatmap of loadings
  output/figures_efa/02_addiction_scores.png      - Per-mouse scores by group
  output/figures_efa/03_factor_scatter.png        - Factor 1 vs Factor 2 scatter
  output/figures_efa/04_stability_histogram.png   - Split-half stability distribution
  output/figures_efa/05_improvement_trajectory.png- Autoresearch improvement
  output/figures_efa/06_group_comparison.png      - Box plot Active vs Passive
  output/figures_efa/07_summary_table.png         - Method comparison table

Usage:
    python visualize_efa.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu

sys.path.insert(0, '.')

OUTPUT = Path(__file__).parent / "output" / "figures_efa"
OUTPUT.mkdir(parents=True, exist_ok=True)

COL_ACTIVE = '#E53935'
COL_PASSIVE = '#1E88E5'


def run_best_pipeline():
    """Run the best pipeline and return full results."""
    from pipeline_efa import run
    return run()


# ── Figure 1: Factor Loadings Heatmap ───────────────────────────────

def plot_loadings_heatmap(result):
    loadings = result['loadings']
    col_names = result['col_names']
    n_factors = loadings.shape[1]

    fig, ax = plt.subplots(figsize=(8, max(6, len(col_names) * 0.35)))

    im = ax.imshow(loadings, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_yticks(range(len(col_names)))
    short_names = [c.replace('_Re-exposure-Pre', '\n(Reexp-Pre)')
                    .replace('_Post-Pre', '\n(Post-Pre)')
                    .replace('_Withdrawal-Pre', '\n(Withdr-Pre)')
                   for c in col_names]
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xticks(range(n_factors))
    ax.set_xticklabels([f'Factor {i+1}' for i in range(n_factors)], fontsize=11)

    for i in range(len(col_names)):
        for j in range(n_factors):
            val = loadings[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold' if abs(val) > 0.4 else 'normal')

    plt.colorbar(im, ax=ax, label='Loading', shrink=0.7)
    ax.set_title('Factor Loadings Heatmap\n(|loading| > 0.4 = strong contribution)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT / '01_factor_loadings.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_factor_loadings.png")


# ── Figure 2: Per-Mouse Addiction Scores ─────────────────────────────

def plot_addiction_scores(result):
    scores = result['scores']
    mice = result['mice']
    groups = result['groups']
    n_factors = scores.shape[1]

    fig, axes = plt.subplots(1, n_factors, figsize=(6 * n_factors, 6))
    if n_factors == 1:
        axes = [axes]

    for f_idx, ax in enumerate(axes):
        active_scores = scores[groups == 'Active', f_idx]
        passive_scores = scores[groups == 'Passive', f_idx]
        active_mice = mice[groups == 'Active']
        passive_mice = mice[groups == 'Passive']

        sorted_idx_a = np.argsort(active_scores)[::-1]
        sorted_idx_p = np.argsort(passive_scores)[::-1]

        all_names = list(active_mice[sorted_idx_a]) + [''] + list(passive_mice[sorted_idx_p])
        all_scores_sorted = list(active_scores[sorted_idx_a]) + [0] + list(passive_scores[sorted_idx_p])
        all_colors = [COL_ACTIVE] * len(active_mice) + ['white'] + [COL_PASSIVE] * len(passive_mice)

        y_pos = range(len(all_names))
        bars = ax.barh(y_pos, all_scores_sorted, color=all_colors, edgecolor='gray', height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_names, fontsize=8)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel(f'Factor {f_idx+1} Score', fontsize=11)
        ax.set_title(f'Factor {f_idx+1}: Per-Mouse Scores', fontsize=12, fontweight='bold')
        ax.invert_yaxis()

        stat, p = mannwhitneyu(active_scores, passive_scores)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'NS'
        ax.text(0.95, 0.02, f'MWU p={p:.4f} {sig}', transform=ax.transAxes,
                fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    active_patch = mpatches.Patch(color=COL_ACTIVE, label='Active')
    passive_patch = mpatches.Patch(color=COL_PASSIVE, label='Passive')
    fig.legend(handles=[active_patch, passive_patch], loc='upper right', fontsize=10)

    fig.suptitle('Addiction Index: Individual Mouse Scores', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT / '02_addiction_scores.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_addiction_scores.png")


# ── Figure 3: Factor 1 vs Factor 2 Scatter ──────────────────────────

def plot_factor_scatter(result):
    scores = result['scores']
    groups = result['groups']
    mice = result['mice']

    if scores.shape[1] < 2:
        print("  Skipping factor scatter (only 1 factor)")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    for group, color, marker in [('Active', COL_ACTIVE, 'o'), ('Passive', COL_PASSIVE, 's')]:
        mask = groups == group
        ax.scatter(scores[mask, 0], scores[mask, 1], c=color, marker=marker,
                   s=120, edgecolors='black', linewidths=0.5, label=group, alpha=0.8, zorder=3)
        for i, m in enumerate(mice[mask]):
            short = m.split('_')[-1] if '_' in m else m[-6:]
            ax.annotate(short, (scores[mask, 0][i], scores[mask, 1][i]),
                        fontsize=7, ha='left', va='bottom', xytext=(4, 4),
                        textcoords='offset points')

    ax.axhline(y=0, color='gray', linewidth=0.3)
    ax.axvline(x=0, color='gray', linewidth=0.3)
    ax.set_xlabel('Factor 1 (Withdrawal-driven)', fontsize=12)
    ax.set_ylabel('Factor 2 (Re-exposure-driven)', fontsize=12)
    ax.set_title('Addiction Index: 2D Factor Space\nActive vs Passive Mice',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.15)

    plt.tight_layout()
    fig.savefig(OUTPUT / '03_factor_scatter.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 03_factor_scatter.png")


# ── Figure 4: Stability Histogram ───────────────────────────────────

def plot_stability_histogram(result):
    from prepare_efa import evaluate_stability
    from sklearn.preprocessing import RobustScaler
    from sklearn.decomposition import FactorAnalysis

    X_raw = result.get('_X_scaled', None)
    if X_raw is None:
        from prepare_efa import load_mouse_level_data
        features = ['RequirementLast', 'lick_freq_per_min', 'bout_n',
                     'rew_n', 'rew_freq_per_min', 'Requirement_speed_per_min']
        delta_pairs = [('Post', 'Pre'), ('Re-exposure', 'Pre'), ('Withdrawal', 'Pre')]
        X_raw_data, _, _, _ = load_mouse_level_data(features=features, delta_pairs=delta_pairs)
        nan_frac = np.mean(np.isnan(X_raw_data), axis=0)
        keep = nan_frac < 0.50
        X_raw_data = X_raw_data[:, keep]
        X = X_raw_data.copy()
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                med = np.nanmedian(X[:, j])
                X[mask, j] = med if np.isfinite(med) else 0.0
        scaler = RobustScaler()
        X_raw = scaler.fit_transform(X)

    n_factors = result['scores'].shape[1]

    def loadings_func(X_sub):
        fa = FactorAnalysis(n_components=n_factors, rotation='quartimax', random_state=42)
        fa.fit(X_sub)
        return fa.components_.T

    _, _, all_corrs = evaluate_stability(loadings_func, X_raw, n_splits=500)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_corrs, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    mean_corr = np.mean(all_corrs)
    ax.axvline(x=mean_corr, color='red', linestyle='--', linewidth=2,
               label=f'Mean = {mean_corr:.3f}')
    ax.axvline(x=0.7, color='green', linestyle=':', linewidth=1.5,
               label='Good threshold (0.7)')
    ax.set_xlabel('Split-Half Loading Correlation', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Factor Stability: Split-Half Cross-Validation (500 splits)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    fig.savefig(OUTPUT / '04_stability_histogram.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_stability_histogram.png")


# ── Figure 5: Autoresearch Improvement Trajectory ────────────────────

def plot_improvement_trajectory():
    experiments = [
        {'exp': 0, 'desc': 'PCA, 48 feats,\n3 factors, Standard', 'quality': 0.414, 'stability': 0.379},
        {'exp': 1, 'desc': '8 feats,\n3 factors, Standard', 'quality': 0.485, 'stability': 0.309},
        {'exp': 2, 'desc': '8 feats,\n2 factors, Robust', 'quality': 0.532, 'stability': 0.489},
        {'exp': 3, 'desc': 'EFA varimax,\n2 factors', 'quality': 0.627, 'stability': 0.509},
        {'exp': 4, 'desc': '+Withdrawal,\n+pharma', 'quality': 0.577, 'stability': 0.523},
        {'exp': 5, 'desc': '3 factors,\n+pharma', 'quality': 0.537, 'stability': 0.398},
        {'exp': 6, 'desc': 'log-transform', 'quality': 0.540, 'stability': 0.457},
        {'exp': 7, 'desc': 'EFA quartimax,\n2 factors', 'quality': 0.663, 'stability': 0.579},
        {'exp': 8, 'desc': '6 core feats,\nquartimax', 'quality': 0.803, 'stability': 0.643},
        {'exp': 9, 'desc': '1 factor only', 'quality': 0.753, 'stability': 0.574},
        {'exp': 10, 'desc': '6 feats,\n+Withdrawal', 'quality': 0.848, 'stability': 0.791},
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    exps = [e['exp'] for e in experiments]
    quals = [e['quality'] for e in experiments]
    stabs = [e['stability'] for e in experiments]
    descs = [e['desc'] for e in experiments]

    best_qual_idx = np.argmax(quals)

    ax1.plot(exps, quals, 'o-', color='#FF7043', linewidth=2, markersize=8, label='Quality Score')
    ax1.scatter([exps[best_qual_idx]], [quals[best_qual_idx]], s=200, color='gold',
                edgecolors='red', zorder=5, marker='*', linewidths=2)
    ax1.set_ylabel('Quality Score', fontsize=12)
    ax1.set_title('Autoresearch Improvement Trajectory: Addiction Index',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.2)
    ax1.axhline(y=0.7, color='green', linestyle=':', alpha=0.4, label='Target')

    ax2.plot(exps, stabs, 's-', color='#42A5F5', linewidth=2, markersize=8, label='Stability')
    ax2.axhline(y=0.7, color='green', linestyle=':', alpha=0.5, label='Good threshold')
    ax2.set_ylabel('Split-Half Stability', fontsize=12)
    ax2.set_xlabel('Experiment #', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.2)

    ax2.set_xticks(exps)
    ax2.set_xticklabels(descs, rotation=45, ha='right', fontsize=7)

    plt.tight_layout()
    fig.savefig(OUTPUT / '05_improvement_trajectory.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_improvement_trajectory.png")


# ── Figure 6: Group Comparison Box Plot ──────────────────────────────

def plot_group_comparison(result):
    scores = result['scores']
    groups = result['groups']
    n_factors = scores.shape[1]

    fig, axes = plt.subplots(1, n_factors, figsize=(5 * n_factors, 6))
    if n_factors == 1:
        axes = [axes]

    for f_idx, ax in enumerate(axes):
        active = scores[groups == 'Active', f_idx]
        passive = scores[groups == 'Passive', f_idx]

        bp = ax.boxplot([active, passive], labels=['Active\n(n=6)', 'Passive\n(n=8)'],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(COL_ACTIVE)
        bp['boxes'][0].set_alpha(0.4)
        bp['boxes'][1].set_facecolor(COL_PASSIVE)
        bp['boxes'][1].set_alpha(0.4)

        jitter_a = np.random.RandomState(42).normal(0, 0.03, len(active))
        jitter_p = np.random.RandomState(43).normal(0, 0.03, len(passive))
        ax.scatter(1 + jitter_a, active, c=COL_ACTIVE, s=60, zorder=5,
                   edgecolors='black', linewidths=0.5)
        ax.scatter(2 + jitter_p, passive, c=COL_PASSIVE, s=60, zorder=5,
                   edgecolors='black', linewidths=0.5)

        stat, p = mannwhitneyu(active, passive)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'NS'
        y_max = max(active.max(), passive.max()) * 1.1
        ax.plot([1, 1, 2, 2], [y_max, y_max * 1.02, y_max * 1.02, y_max], color='black', linewidth=1)
        ax.text(1.5, y_max * 1.04, f'{sig}\np={p:.4f}', ha='center', fontsize=10, fontweight='bold')

        factor_labels = {0: 'Withdrawal Response', 1: 'Re-exposure Response'}
        ax.set_ylabel('Factor Score', fontsize=11)
        ax.set_title(f'Factor {f_idx+1}: {factor_labels.get(f_idx, "")}',
                     fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)

    fig.suptitle('Addiction Index: Active vs Passive Group Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT / '06_group_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 06_group_comparison.png")


# ── Figure 7: Summary Comparison Table ───────────────────────────────

def plot_summary_table():
    experiments = [
        ['Exp 0 (Baseline)', 'PCA', 'StandardScaler', '48 (all)', '3', 'none', '0.379', '0.748', '0', '0.414'],
        ['Exp 3 (Varimax)', 'EFA', 'RobustScaler', '16 (8 feats)', '2', 'varimax', '0.509', '0.743', '12', '0.627'],
        ['Exp 7 (Quartimax)', 'EFA', 'RobustScaler', '16 (8 feats)', '2', 'quartimax', '0.579', '0.743', '12', '0.663'],
        ['Exp 8 (Core)', 'EFA', 'RobustScaler', '12 (6 feats)', '2', 'quartimax', '0.643', '0.938', '12', '0.803'],
        ['Exp 10 (BEST)', 'EFA', 'RobustScaler', '18 (6f x 3d)', '2', 'quartimax', '0.791', '0.841', '18', '0.848'],
    ]
    columns = ['Experiment', 'Method', 'Scaler', 'Features', 'Factors', 'Rotation',
               'Stability', 'Var Expl.', 'High Load', 'Quality']

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis('off')

    table = ax.table(cellText=experiments, colLabels=columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for j in range(len(columns)):
        table[0, j].set_facecolor('#2196F3')
        table[0, j].set_text_props(color='white', fontweight='bold')

    for j in range(len(columns)):
        table[len(experiments), j].set_facecolor('#E8F5E9')
        table[len(experiments), j].set_text_props(fontweight='bold')

    ax.set_title('Autoresearch Experiment Comparison: Addiction Index',
                 fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    fig.savefig(OUTPUT / '07_summary_table.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 07_summary_table.png")


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ADDICTION INDEX (EFA) VISUALIZATION")
    print("=" * 60)

    print("\nRunning best pipeline...")
    result = run_best_pipeline()

    print("\n[1] Factor loadings heatmap...")
    plot_loadings_heatmap(result)

    print("\n[2] Per-mouse addiction scores...")
    plot_addiction_scores(result)

    print("\n[3] Factor scatter (2D)...")
    plot_factor_scatter(result)

    print("\n[4] Stability histogram...")
    plot_stability_histogram(result)

    print("\n[5] Improvement trajectory...")
    plot_improvement_trajectory()

    print("\n[6] Group comparison box plot...")
    plot_group_comparison(result)

    print("\n[7] Summary table...")
    plot_summary_table()

    print(f"\nAll figures saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
