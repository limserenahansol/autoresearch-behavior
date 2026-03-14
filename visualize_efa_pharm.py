"""
visualize_efa_pharm.py  --  EFA with Pharmacological Data Visualization
========================================================================
Same figures as visualize_efa.py but for the behavioral+pharma model.

Output: output/figures_efa_pharm/
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

OUTPUT = Path(__file__).parent / "output" / "figures_efa_pharm"
OUTPUT.mkdir(parents=True, exist_ok=True)

COL_ACTIVE = '#E53935'
COL_PASSIVE = '#1E88E5'


def run_pipeline():
    from pipeline_efa_pharm import run
    return run()


def shorten(name):
    name = (name.replace('Requirement_speed_per_min', 'Req_speed')
                .replace('RequirementLast', 'ReqLast')
                .replace('lick_freq_per_min', 'lick_freq')
                .replace('rew_freq_per_min', 'rew_freq')
                .replace('lick_meanDur_s', 'lick_dur')
                .replace('lick_totalDur_s', 'lick_totDur')
                .replace('Immersion_Latency_s', 'TI_latency')
                .replace('TST_Pct_Non_moving', 'TST_still')
                .replace('TST_Pct_Licking', 'TST_lick')
                .replace('TST_Pct_Rearing', 'TST_rear')
                .replace('HOT_Pct_Non_moving', 'HP_still')
                .replace('HOT_Pct_Licking', 'HP_lick')
                .replace('HOT_Pct_Rearing', 'HP_rear')
                .replace('pupil_mean', 'pupil')
                .replace('_Re-exposure-Pre', '\n(R-Pre)')
                .replace('_Post-Pre', '\n(P-Pre)')
                .replace('_Withdrawal-Pre', '\n(W-Pre)'))
    return name


# ── Figure 1: Factor Loadings Heatmap ───────────────────────────────

def plot_loadings_heatmap(result):
    loadings = result['loadings']
    col_names = result['col_names']
    n_factors = loadings.shape[1]

    short_names = [shorten(c) for c in col_names]

    fig, ax = plt.subplots(figsize=(10, max(8, len(col_names) * 0.32)))

    im = ax.imshow(loadings, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=6)
    ax.set_xticks(range(n_factors))
    factor_labels = ['Factor 1\n(Withdrawal)', 'Factor 2\n(Re-exposure)', 'Factor 3\n(Pharmacological)']
    ax.set_xticklabels(factor_labels[:n_factors], fontsize=10)

    for i in range(len(col_names)):
        for j in range(n_factors):
            val = loadings[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=5, color=color, fontweight='bold' if abs(val) > 0.4 else 'normal')

    # Color-code rows by variable type
    for i, c in enumerate(col_names):
        if any(p in c for p in ['TST_', 'HOT_', 'Immersion']):
            ax.get_yticklabels()[i].set_color('#9C27B0')
        elif 'pupil' in c:
            ax.get_yticklabels()[i].set_color('#00897B')

    plt.colorbar(im, ax=ax, label='Loading', shrink=0.5)
    ax.set_title('Factor Loadings: Behavioral + Pharmacological Variables\n'
                 '(purple = pharma, teal = pupil, black = behavioral)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT / '01_factor_loadings_pharm.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_factor_loadings_pharm.png")


# ── Figure 2: Factor Scatter ────────────────────────────────────────

def plot_factor_scatter(result):
    scores = result['scores']
    groups = result['groups']
    mice = result['mice']
    n_factors = scores.shape[1]

    if n_factors < 2:
        return

    n_pairs = min(3, n_factors * (n_factors - 1) // 2)
    pairs = [(0, 1), (0, 2), (1, 2)][:n_pairs]

    fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 6))
    if n_pairs == 1:
        axes = [axes]

    factor_names = ['F1 (Withdrawal)', 'F2 (Re-exposure)', 'F3 (Pharmacological)']

    for ax, (fi, fj) in zip(axes, pairs):
        for group, color, marker in [('Active', COL_ACTIVE, 'o'), ('Passive', COL_PASSIVE, 's')]:
            mask = groups == group
            ax.scatter(scores[mask, fi], scores[mask, fj], c=color, marker=marker,
                       s=100, edgecolors='black', linewidths=0.5, label=group, alpha=0.8, zorder=3)
            for i, m in enumerate(mice[mask]):
                short = m.split('_')[-1] if '_' in m else m[-6:]
                ax.annotate(short, (scores[mask, fi][i], scores[mask, fj][i]),
                            fontsize=6, ha='left', va='bottom', xytext=(3, 3),
                            textcoords='offset points')
        ax.axhline(0, color='gray', linewidth=0.3)
        ax.axvline(0, color='gray', linewidth=0.3)
        ax.set_xlabel(factor_names[fi], fontsize=10)
        ax.set_ylabel(factor_names[fj], fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.15)

    fig.suptitle('Addiction Index with Pharma: Factor Space',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT / '02_factor_scatter_pharm.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_factor_scatter_pharm.png")


# ── Figure 3: Group Comparison Box Plot ──────────────────────────────

def plot_group_comparison(result):
    scores = result['scores']
    groups = result['groups']
    n_factors = scores.shape[1]
    group_sep_p = result['group_sep_p']

    factor_names = ['Factor 1:\nWithdrawal Response', 'Factor 2:\nRe-exposure Response',
                    'Factor 3:\nPharmacological Response']

    fig, axes = plt.subplots(1, n_factors, figsize=(5 * n_factors, 6))
    if n_factors == 1:
        axes = [axes]

    for f_idx, ax in enumerate(axes):
        active = scores[groups == 'Active', f_idx]
        passive = scores[groups == 'Passive', f_idx]

        bp = ax.boxplot([active, passive], tick_labels=['Active\n(n=6)', 'Passive\n(n=8)'],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(COL_ACTIVE)
        bp['boxes'][0].set_alpha(0.4)
        bp['boxes'][1].set_facecolor(COL_PASSIVE)
        bp['boxes'][1].set_alpha(0.4)

        rng = np.random.RandomState(42 + f_idx)
        ax.scatter(1 + rng.normal(0, 0.03, len(active)), active,
                   c=COL_ACTIVE, s=60, zorder=5, edgecolors='black', linewidths=0.5)
        ax.scatter(2 + rng.normal(0, 0.03, len(passive)), passive,
                   c=COL_PASSIVE, s=60, zorder=5, edgecolors='black', linewidths=0.5)

        p = group_sep_p[f_idx]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'NS'
        y_max = max(active.max(), passive.max()) * 1.15
        ax.plot([1, 1, 2, 2], [y_max, y_max * 1.02, y_max * 1.02, y_max], color='black', linewidth=1)
        ax.text(1.5, y_max * 1.04, f'{sig}\np={p:.4f}', ha='center', fontsize=10, fontweight='bold')

        ax.set_ylabel('Factor Score', fontsize=11)
        ax.set_title(factor_names[f_idx], fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)

    fig.suptitle('Addiction Index (with Pharma): Active vs Passive',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT / '03_group_comparison_pharm.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 03_group_comparison_pharm.png")


# ── Figure 4: Side-by-side comparison with behavioral-only ──────────

def plot_comparison_table(result):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')

    data = [
        ['Behavioral Only (BEST)', '6', '18', '2', 'quartimax',
         '0.791', '0.841', '18/18', '0.848', 'p=0.008**'],
        ['Behavioral + Pharma', '14', str(len(result['col_names'])), '3', 'quartimax',
         f'{result["stability"]:.3f}', f'{result["var_explained"]:.3f}',
         f'{result["n_high_loading"]}/{len(result["col_names"])}',
         f'{result["quality_score"]:.3f}',
         f'p={min(result["group_sep_p"]):.4f}{"**" if min(result["group_sep_p"])<0.01 else "*" if min(result["group_sep_p"])<0.05 else " NS"}'],
    ]
    columns = ['Model', 'Variables', 'Delta\nFeatures', 'Factors', 'Rotation',
               'Stability', 'Var Expl.', 'High\nLoadings', 'Quality', 'Best Group\nSeparation']

    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    for j in range(len(columns)):
        table[0, j].set_facecolor('#2196F3')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for j in range(len(columns)):
        table[1, j].set_facecolor('#E8F5E9')
        table[1, j].set_text_props(fontweight='bold')
    for j in range(len(columns)):
        table[2, j].set_facecolor('#FFF3E0')

    ax.set_title('Comparison: Behavioral Only vs Behavioral + Pharmacological',
                 fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    fig.savefig(OUTPUT / '04_comparison_table.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_comparison_table.png")


# ── Figure 5: Per-mouse scores ───────────────────────────────────────

def plot_addiction_scores(result):
    scores = result['scores']
    mice = result['mice']
    groups = result['groups']
    n_factors = scores.shape[1]

    fig, axes = plt.subplots(1, n_factors, figsize=(6 * n_factors, 6))
    if n_factors == 1:
        axes = [axes]

    factor_names = ['F1: Withdrawal', 'F2: Re-exposure', 'F3: Pharmacological']

    for f_idx, ax in enumerate(axes):
        active_scores = scores[groups == 'Active', f_idx]
        passive_scores = scores[groups == 'Passive', f_idx]
        active_mice = mice[groups == 'Active']
        passive_mice = mice[groups == 'Passive']

        sorted_a = np.argsort(active_scores)[::-1]
        sorted_p = np.argsort(passive_scores)[::-1]

        all_names = list(active_mice[sorted_a]) + [''] + list(passive_mice[sorted_p])
        all_vals = list(active_scores[sorted_a]) + [0] + list(passive_scores[sorted_p])
        all_colors = [COL_ACTIVE] * len(active_mice) + ['white'] + [COL_PASSIVE] * len(passive_mice)

        ax.barh(range(len(all_names)), all_vals, color=all_colors, edgecolor='gray', height=0.7)
        ax.set_yticks(range(len(all_names)))
        ax.set_yticklabels(all_names, fontsize=7)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel(f'{factor_names[f_idx]} Score', fontsize=10)
        ax.set_title(factor_names[f_idx], fontsize=11, fontweight='bold')
        ax.invert_yaxis()

        p = result['group_sep_p'][f_idx]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'NS'
        ax.text(0.95, 0.02, f'MWU p={p:.4f} {sig}', transform=ax.transAxes,
                fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Addiction Index (with Pharma): Individual Mouse Scores',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT / '05_addiction_scores_pharm.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_addiction_scores_pharm.png")


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EFA WITH PHARMACOLOGICAL DATA")
    print("=" * 60)

    print("\nRunning pipeline with pharma...")
    result = run_pipeline()

    print("\n[1] Factor loadings heatmap...")
    plot_loadings_heatmap(result)

    print("\n[2] Factor scatter...")
    plot_factor_scatter(result)

    print("\n[3] Group comparison...")
    plot_group_comparison(result)

    print("\n[4] Comparison table...")
    plot_comparison_table(result)

    print("\n[5] Per-mouse scores...")
    plot_addiction_scores(result)

    print(f"\nAll figures saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
