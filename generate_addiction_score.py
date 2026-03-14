"""
generate_addiction_score.py  --  Final Addiction Score Output
=============================================================
The actual "result" figure: each mouse's addiction score,
combining Factor 1 and Factor 2 into a single composite index,
plus individual factor scores and a ranking.

Output:
  output/figures_efa/08_addiction_score_final.png
  output/figures_efa/09_addiction_score_table.png
  output/addiction_scores.csv
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

OUTPUT_FIG = Path(__file__).parent / "output" / "figures_efa"
OUTPUT = Path(__file__).parent / "output"
OUTPUT_FIG.mkdir(parents=True, exist_ok=True)

COL_ACTIVE = '#E53935'
COL_PASSIVE = '#1E88E5'


def run_pipeline():
    from pipeline_efa import run
    return run()


def main():
    print("Running best EFA pipeline...")
    result = run_pipeline()

    scores = result['scores']
    mice = result['mice']
    groups = result['groups']
    loadings = result['loadings']
    col_names = result['col_names']

    f1 = scores[:, 0]
    f2 = scores[:, 1]

    # Composite addiction score: weighted by variance explained
    model = result['model']
    total_var = np.sum(np.var(scores, axis=0))
    var_per_factor = np.var(scores, axis=0)
    weights = var_per_factor / total_var
    composite = weights[0] * f1 + weights[1] * f2

    # Normalize composite to 0-100 scale
    composite_norm = (composite - composite.min()) / (composite.max() - composite.min()) * 100

    # Save CSV
    df_out = pd.DataFrame({
        'mouse': mice,
        'group': groups,
        'factor1_withdrawal': np.round(f1, 4),
        'factor2_reexposure': np.round(f2, 4),
        'composite_raw': np.round(composite, 4),
        'addiction_score_0to100': np.round(composite_norm, 1),
    })
    df_out = df_out.sort_values('addiction_score_0to100', ascending=False).reset_index(drop=True)
    df_out.index.name = 'rank'
    df_out.to_csv(OUTPUT / 'addiction_scores.csv', index=True)
    print(f"  Saved: addiction_scores.csv")

    # ═══════════════════════════════════════════════════════════════
    # FIGURE 1: Comprehensive Addiction Score Dashboard
    # ═══════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(24, 14))

    # ── Panel A: Composite Addiction Score (horizontal bar, ranked) ──
    ax_a = fig.add_axes([0.04, 0.08, 0.28, 0.82])

    sort_idx = np.argsort(composite_norm)[::-1]
    sorted_mice = mice[sort_idx]
    sorted_scores = composite_norm[sort_idx]
    sorted_groups = groups[sort_idx]
    bar_colors = [COL_ACTIVE if g == 'Active' else COL_PASSIVE for g in sorted_groups]

    y_pos = np.arange(len(sorted_mice))
    bars = ax_a.barh(y_pos, sorted_scores, color=bar_colors, edgecolor='white', height=0.7)

    for i, (bar, val, grp) in enumerate(zip(bars, sorted_scores, sorted_groups)):
        ax_a.text(val + 1.5, i, f'{val:.0f}', va='center', fontsize=12, fontweight='bold',
                  color=COL_ACTIVE if grp == 'Active' else COL_PASSIVE)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(sorted_mice, fontsize=12, fontweight='bold')
    ax_a.set_xlabel('Addiction Score (0-100)', fontsize=14, fontweight='bold')
    ax_a.set_title('A. Addiction Score Ranking\n(higher = more addiction-like behavior)',
                   fontsize=16, fontweight='bold', pad=15)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 115)
    ax_a.axvline(x=50, color='gray', linestyle=':', alpha=0.4)
    ax_a.grid(axis='x', alpha=0.15)

    active_patch = mpatches.Patch(color=COL_ACTIVE, label='Active (n=6)')
    passive_patch = mpatches.Patch(color=COL_PASSIVE, label='Passive (n=8)')
    ax_a.legend(handles=[active_patch, passive_patch], fontsize=12, loc='lower right')

    # Group stats
    active_scores = composite_norm[groups == 'Active']
    passive_scores = composite_norm[groups == 'Passive']
    stat, p = mannwhitneyu(active_scores, passive_scores)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'NS'
    ax_a.text(0.95, 0.02,
              f'Active mean: {active_scores.mean():.0f}\n'
              f'Passive mean: {passive_scores.mean():.0f}\n'
              f'MWU p={p:.4f} {sig}',
              transform=ax_a.transAxes, ha='right', va='bottom', fontsize=11,
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))

    # ── Panel B: Factor 1 vs Factor 2 with composite as color ───────
    ax_b = fig.add_axes([0.38, 0.52, 0.28, 0.40])

    sc = ax_b.scatter(f1, f2, c=composite_norm, cmap='YlOrRd', s=200,
                       edgecolors='black', linewidths=1.5, zorder=3, vmin=0, vmax=100)
    for i, m in enumerate(mice):
        short = m.split('_')[-1] if '_' in m else m[-6:]
        marker_style = 'o' if groups[i] == 'Active' else 's'
        ax_b.scatter(f1[i], f2[i], c=[composite_norm[i]], cmap='YlOrRd',
                     marker=marker_style, s=200, edgecolors='black', linewidths=1.5,
                     zorder=4, vmin=0, vmax=100)
        ax_b.annotate(short, (f1[i], f2[i]), fontsize=10, fontweight='bold',
                      xytext=(6, 6), textcoords='offset points')

    ax_b.axhline(0, color='gray', linewidth=0.5)
    ax_b.axvline(0, color='gray', linewidth=0.5)
    ax_b.set_xlabel('Factor 1: Withdrawal Response', fontsize=13, fontweight='bold')
    ax_b.set_ylabel('Factor 2: Re-exposure Response', fontsize=13, fontweight='bold')
    ax_b.set_title('B. 2D Addiction Space\n(color = composite score)',
                   fontsize=16, fontweight='bold', pad=12)
    ax_b.grid(alpha=0.15)
    ax_b.tick_params(labelsize=11)
    cb = plt.colorbar(sc, ax=ax_b, shrink=0.8, pad=0.02)
    cb.set_label('Addiction Score (0-100)', fontsize=11)

    # ── Panel C: Group comparison box plot ───────────────────────────
    ax_c = fig.add_axes([0.72, 0.52, 0.25, 0.40])

    bp = ax_c.boxplot([active_scores, passive_scores],
                       tick_labels=['Active\n(n=6)', 'Passive\n(n=8)'],
                       patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(COL_ACTIVE)
    bp['boxes'][0].set_alpha(0.35)
    bp['boxes'][1].set_facecolor(COL_PASSIVE)
    bp['boxes'][1].set_alpha(0.35)

    rng = np.random.RandomState(42)
    ax_c.scatter(1 + rng.normal(0, 0.04, len(active_scores)), active_scores,
                 c=COL_ACTIVE, s=100, zorder=5, edgecolors='black', linewidths=0.8)
    ax_c.scatter(2 + rng.normal(0, 0.04, len(passive_scores)), passive_scores,
                 c=COL_PASSIVE, s=100, zorder=5, edgecolors='black', linewidths=0.8)

    y_max = max(active_scores.max(), passive_scores.max()) + 5
    ax_c.plot([1, 1, 2, 2], [y_max, y_max + 2, y_max + 2, y_max], color='black', linewidth=1.5)
    ax_c.text(1.5, y_max + 3, f'{sig}\np={p:.4f}', ha='center', fontsize=14, fontweight='bold')

    ax_c.set_ylabel('Addiction Score (0-100)', fontsize=13, fontweight='bold')
    ax_c.set_title('C. Active vs Passive\nComposite Score',
                   fontsize=16, fontweight='bold', pad=12)
    ax_c.set_ylim(-5, 115)
    ax_c.grid(axis='y', alpha=0.2)
    ax_c.tick_params(labelsize=12)

    # ── Panel D: How composite is calculated ─────────────────────────
    ax_d = fig.add_axes([0.38, 0.08, 0.58, 0.35])
    ax_d.axis('off')
    ax_d.set_xlim(0, 10)
    ax_d.set_ylim(0, 5)

    ax_d.text(5, 4.7, 'D. How the Addiction Score is Calculated',
              fontsize=16, fontweight='bold', ha='center', va='top', color='#0D47A1')

    explanation = (
        f"Addiction Score = {weights[0]:.2f} x Factor1 (Withdrawal) + {weights[1]:.2f} x Factor2 (Re-exposure)\n"
        f"                   then scaled to 0-100 range\n\n"
        f"Factor 1 captures: how behavior changes during WITHDRAWAL (lick_freq, bout_n, PR breakpoint decline)\n"
        f"Factor 2 captures: how behavior changes during RE-EXPOSURE (PR breakpoint, reward rate increase)\n\n"
        f"Weights are proportional to variance explained by each factor.\n"
        f"Factor 2 (Re-exposure) has higher weight because it explains more variance ({weights[1]:.0%} vs {weights[0]:.0%}).\n\n"
        f"Interpretation:\n"
        f"  Score > 70: Strong addiction-like profile (high drug-seeking, resilient to withdrawal)\n"
        f"  Score 30-70: Moderate response\n"
        f"  Score < 30: Low addiction-like profile (low drug-seeking, sensitive to withdrawal)"
    )
    ax_d.text(0.2, 3.5, explanation, fontsize=11, va='top', family='monospace',
              bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F5F5',
                        edgecolor='#999', linewidth=1))

    fig.suptitle('Addiction Index: Final Scores per Mouse',
                 fontsize=22, fontweight='bold', color='#0D47A1', y=0.98)

    fig.savefig(OUTPUT_FIG / '08_addiction_score_final.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 08_addiction_score_final.png")

    # ═══════════════════════════════════════════════════════════════
    # FIGURE 2: Score Table
    # ═══════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    ax2.axis('off')

    table_data = []
    for _, row in df_out.iterrows():
        table_data.append([
            row['mouse'],
            row['group'],
            f"{row['factor1_withdrawal']:.2f}",
            f"{row['factor2_reexposure']:.2f}",
            f"{row['addiction_score_0to100']:.0f}",
            'High' if row['addiction_score_0to100'] > 70 else
            'Moderate' if row['addiction_score_0to100'] > 30 else 'Low',
        ])

    columns = ['Mouse', 'Group', 'Factor 1\n(Withdrawal)', 'Factor 2\n(Re-exposure)',
               'Addiction\nScore', 'Profile']

    table = ax2.table(cellText=table_data, colLabels=columns,
                       loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    for j in range(len(columns)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=12)

    for i, row in enumerate(table_data):
        grp = row[1]
        score = float(row[4])
        bg_color = '#FFEBEE' if grp == 'Active' else '#E3F2FD'
        for j in range(len(columns)):
            table[i + 1, j].set_facecolor(bg_color)
        if score > 70:
            table[i + 1, 5].set_text_props(color='#C62828', fontweight='bold')
        elif score < 30:
            table[i + 1, 5].set_text_props(color='#1565C0', fontweight='bold')

    ax2.set_title('Addiction Score: Complete Results Table\n'
                  '(ranked from highest to lowest)',
                  fontsize=16, fontweight='bold', pad=20, color='#0D47A1')

    fig2.savefig(OUTPUT_FIG / '09_addiction_score_table.png', dpi=200,
                 bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 09_addiction_score_table.png")

    # Print summary
    print("\n" + "=" * 50)
    print("ADDICTION SCORE SUMMARY")
    print("=" * 50)
    print(df_out.to_string(index=True))
    print(f"\nActive mean:  {active_scores.mean():.1f}")
    print(f"Passive mean: {passive_scores.mean():.1f}")
    print(f"MWU p = {p:.4f} {sig}")


if __name__ == "__main__":
    main()
