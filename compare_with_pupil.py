"""
compare_with_pupil.py
======================
Generates comparison figures:
  1. Decoder comparison (bar chart: original vs +pupil for all 3 tasks)
  2. EFA comparison (quality, stability, var_explained side by side)
  3. EFA loadings heatmap for +pupil model
  4. EFA factor scatter for +pupil model
  5. EFA group comparison for +pupil model
  6. Decoder confusion matrices for +pupil model
  7. Combined summary table

All saved to: output/with_pupil/figures/
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from scipy.stats import mannwhitneyu
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, str(Path(__file__).parent))

OUTDIR = Path(__file__).parent / "output" / "with_pupil" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

COL_ORIG = '#42A5F5'
COL_PUPIL = '#FF7043'
COL_ACTIVE = '#E53935'
COL_PASSIVE = '#1E88E5'

ORIGINAL_DECODER = {
    'Period (5-class)':    {'per_mouse_acc': 0.937500, 'accuracy': 0.936073, 'f1_macro': 0.921722},
    'Substance (2-class)': {'per_mouse_acc': 0.933036, 'accuracy': 0.931507, 'f1_macro': 0.925539},
    'Group (2-class)':     {'per_mouse_acc': 0.788555, 'accuracy': 0.789954, 'f1_macro': 0.785660},
}

ORIGINAL_EFA = {
    'stability': 0.791, 'var_explained': 0.841, 'quality_score': 0.848,
    'n_high_loading': 18, 'n_features': 18,
}


def load_new_decoder_summary():
    path = Path(__file__).parent / "output" / "with_pupil" / "summary_all_classifiers.csv"
    df = pd.read_csv(path)
    result = {}
    for _, row in df.iterrows():
        result[row['task']] = {
            'per_mouse_acc': row['per_mouse_acc'],
            'accuracy': row['accuracy'],
            'f1_macro': row['f1_macro'],
        }
    return result


# ═════════════════════════════════════════════════════════════════════
# Figure 1: Decoder Comparison Bar Chart
# ═════════════════════════════════════════════════════════════════════
def plot_decoder_comparison():
    new = load_new_decoder_summary()
    tasks = ['Period (5-class)', 'Substance (2-class)', 'Group (2-class)']
    metrics = ['per_mouse_acc', 'accuracy', 'f1_macro']
    metric_labels = ['Per-Mouse Acc', 'Accuracy', 'F1 Macro']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        orig_vals = [ORIGINAL_DECODER[t][metric] for t in tasks]
        new_vals = [new[t][metric] for t in tasks]

        x = np.arange(len(tasks))
        w = 0.32
        bars1 = ax.bar(x - w/2, orig_vals, w, label='Without Pupil Peak',
                        color=COL_ORIG, edgecolor='white')
        bars2 = ax.bar(x + w/2, new_vals, w, label='+ pupil_reward_peak',
                        color=COL_PUPIL, edgecolor='white')

        for b, v in zip(bars1, orig_vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                    f'{v:.3f}', ha='center', fontsize=8, color=COL_ORIG, fontweight='bold')
        for b, v in zip(bars2, new_vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                    f'{v:.3f}', ha='center', fontsize=8, color=COL_PUPIL, fontweight='bold')

        for i in range(len(tasks)):
            delta = new_vals[i] - orig_vals[i]
            sign = '+' if delta >= 0 else ''
            color = '#4CAF50' if delta >= 0 else '#F44336'
            ax.text(x[i], max(orig_vals[i], new_vals[i]) + 0.02,
                    f'{sign}{delta:.4f}', ha='center', fontsize=9,
                    color=color, fontweight='bold')

        short_tasks = ['Period\n(5-class)', 'Substance\n(2-class)', 'Group\n(2-class)']
        ax.set_xticks(x)
        ax.set_xticklabels(short_tasks, fontsize=9)
        ax.set_ylabel(mlabel, fontsize=11)
        ax.set_ylim(0.6, 1.05)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(axis='y', alpha=0.2)
        ax.set_title(mlabel, fontsize=13, fontweight='bold')

    fig.suptitle('Decoder Comparison: Adding Peak Reward-Locked Pupil Dilation',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTDIR / '01_decoder_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_decoder_comparison.png")


# ═════════════════════════════════════════════════════════════════════
# Figure 2: EFA Comparison
# ═════════════════════════════════════════════════════════════════════
def plot_efa_comparison(new_efa):
    metrics = ['quality_score', 'stability', 'var_explained']
    labels = ['Quality Score', 'Stability', 'Variance Explained']

    orig_vals = [ORIGINAL_EFA[m] for m in metrics]
    new_vals = [new_efa[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    w = 0.32

    bars1 = ax.bar(x - w/2, orig_vals, w, label='Without Pupil Peak (6 features)',
                    color=COL_ORIG, edgecolor='white')
    bars2 = ax.bar(x + w/2, new_vals, w, label='+ pupil_reward_peak (7 features)',
                    color=COL_PUPIL, edgecolor='white')

    for b, v in zip(bars1, orig_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01,
                f'{v:.3f}', ha='center', fontsize=10, color=COL_ORIG, fontweight='bold')
    for b, v in zip(bars2, new_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01,
                f'{v:.3f}', ha='center', fontsize=10, color=COL_PUPIL, fontweight='bold')

    for i in range(len(metrics)):
        delta = new_vals[i] - orig_vals[i]
        sign = '+' if delta >= 0 else ''
        color = '#4CAF50' if delta >= 0 else '#F44336'
        ax.text(x[i], max(orig_vals[i], new_vals[i]) + 0.04,
                f'{sign}{delta:.3f}', ha='center', fontsize=11,
                color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.2)
    ax.set_title('EFA Addiction Index Comparison:\nAdding Peak Reward-Locked Pupil Dilation',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTDIR / '02_efa_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_efa_comparison.png")


# ═════════════════════════════════════════════════════════════════════
# Figure 3: New EFA Loadings Heatmap
# ═════════════════════════════════════════════════════════════════════
def plot_loadings(result):
    loadings = result['loadings']
    col_names = result['col_names']
    n_factors = loadings.shape[1]

    fig, ax = plt.subplots(figsize=(9, max(7, len(col_names) * 0.38)))
    im = ax.imshow(loadings, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_yticks(range(len(col_names)))
    short = [c.replace('_Re-exposure-Pre', '\n(Reexp-Pre)')
              .replace('_Post-Pre', '\n(Post-Pre)')
              .replace('_Withdrawal-Pre', '\n(Withdr-Pre)')
             for c in col_names]
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xticks(range(n_factors))
    ax.set_xticklabels([f'Factor {i+1}' for i in range(n_factors)], fontsize=12)

    for i in range(len(col_names)):
        for j in range(n_factors):
            val = loadings[i, j]
            is_pupil = 'pupil_reward_peak' in col_names[i]
            color = 'white' if abs(val) > 0.5 else 'black'
            weight = 'bold' if abs(val) > 0.4 or is_pupil else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color, fontweight=weight)
            if is_pupil:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor='gold', linewidth=2.5))

    plt.colorbar(im, ax=ax, label='Loading', shrink=0.6)
    ax.set_title('Factor Loadings: EFA + pupil_reward_peak\n(gold border = pupil feature)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTDIR / '03_efa_loadings_with_pupil.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 03_efa_loadings_with_pupil.png")


# ═════════════════════════════════════════════════════════════════════
# Figure 4: Factor Scatter
# ═════════════════════════════════════════════════════════════════════
def plot_factor_scatter(result):
    scores = result['scores']
    groups = result['groups']
    mice = result['mice']

    if scores.shape[1] < 2:
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
    ax.set_xlabel('Factor 1', fontsize=12)
    ax.set_ylabel('Factor 2', fontsize=12)
    ax.set_title('Addiction Index + Pupil: 2D Factor Space',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.15)

    for f_idx in range(2):
        p = result['group_sep_p'][f_idx]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'NS'
        ax.text(0.02, 0.98 - f_idx * 0.05,
                f'F{f_idx+1} group: p={p:.4f} {sig}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig.savefig(OUTDIR / '04_efa_scatter_with_pupil.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_efa_scatter_with_pupil.png")


# ═════════════════════════════════════════════════════════════════════
# Figure 5: Group Comparison Box Plot
# ═════════════════════════════════════════════════════════════════════
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
        ymax = max(active.max(), passive.max())
        ymin = min(active.min(), passive.min())
        yrange = ymax - ymin
        bar_y = ymax + yrange * 0.1
        ax.plot([1, 1, 2, 2], [bar_y, bar_y + yrange*0.02, bar_y + yrange*0.02, bar_y],
                color='black', linewidth=1)
        ax.text(1.5, bar_y + yrange * 0.05, f'{sig}\np={p:.4f}',
                ha='center', fontsize=10, fontweight='bold')

        ax.set_ylabel('Factor Score', fontsize=11)
        ax.set_title(f'Factor {f_idx+1}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)

    fig.suptitle('EFA + Pupil: Active vs Passive Group Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTDIR / '05_efa_group_with_pupil.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_efa_group_with_pupil.png")


# ═════════════════════════════════════════════════════════════════════
# Figure 6: Decoder Confusion Matrices (with pupil)
# ═════════════════════════════════════════════════════════════════════
def plot_confusion_matrices():
    tasks = [
        ('5class', ['Pre', 'During', 'Post', 'Withdrawal', 'Re-exposure'], 'Period (5-class)'),
        ('2class', ['morphine', 'water'], 'Substance (2-class)'),
        ('group', ['Active', 'Passive'], 'Group (2-class)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    wp = Path(__file__).parent / "output" / "with_pupil"

    for ax, (task, classes, title) in zip(axes, tasks):
        pred_path = wp / f"predictions_{task}.csv"
        if not pred_path.exists():
            continue
        df = pd.read_csv(pred_path)
        cm = confusion_matrix(df['y_true'], df['y_pred'], labels=classes)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        disp = ConfusionMatrixDisplay(cm_pct, display_labels=classes)
        disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='.1f')
        overall = np.trace(cm) / cm.sum()
        ax.set_title(f'{title}\nacc={overall:.1%}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Actual', fontsize=9)
        for t in ax.texts:
            t.set_fontsize(8)

    fig.suptitle('Confusion Matrices (+ pupil_reward_peak)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTDIR / '06_confusion_with_pupil.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 06_confusion_with_pupil.png")


# ═════════════════════════════════════════════════════════════════════
# Figure 7: Combined Summary Table
# ═════════════════════════════════════════════════════════════════════
def plot_summary_table(new_efa):
    new_dec = load_new_decoder_summary()

    rows = []
    for task in ['Period (5-class)', 'Substance (2-class)', 'Group (2-class)']:
        o = ORIGINAL_DECODER[task]
        n = new_dec[task]
        delta = n['per_mouse_acc'] - o['per_mouse_acc']
        sign = '+' if delta >= 0 else ''
        rows.append([
            f'Decoder: {task}',
            f"{o['per_mouse_acc']:.4f}",
            f"{n['per_mouse_acc']:.4f}",
            f"{sign}{delta:.4f}",
        ])

    for metric, label in [('quality_score', 'EFA Quality'), ('stability', 'EFA Stability'),
                            ('var_explained', 'EFA Var Expl.')]:
        o_val = ORIGINAL_EFA[metric]
        n_val = new_efa[metric]
        delta = n_val - o_val
        sign = '+' if delta >= 0 else ''
        rows.append([label, f"{o_val:.3f}", f"{n_val:.3f}", f"{sign}{delta:.3f}"])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    cols = ['Metric', 'Without Pupil', '+ Pupil Peak', 'Change']
    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    for j in range(len(cols)):
        table[0, j].set_facecolor('#2196F3')
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=11)

    for i, row in enumerate(rows, start=1):
        delta_str = row[3]
        if delta_str.startswith('+'):
            table[i, 3].set_text_props(color='#2E7D32', fontweight='bold')
        elif delta_str.startswith('-'):
            table[i, 3].set_text_props(color='#C62828', fontweight='bold')

    ax.set_title('Summary: Impact of Adding pupil_reward_peak',
                 fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    fig.savefig(OUTDIR / '07_summary_table.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 07_summary_table.png")


# ═════════════════════════════════════════════════════════════════════
# Figure 8: Stability histogram for +pupil EFA
# ═════════════════════════════════════════════════════════════════════
def plot_stability_histogram(result):
    all_corrs = result.get('all_corrs', [])
    if not all_corrs:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_corrs, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    mean_c = np.mean(all_corrs)
    ax.axvline(x=mean_c, color='red', linestyle='--', linewidth=2,
               label=f'Mean = {mean_c:.3f}')
    ax.axvline(x=0.7, color='green', linestyle=':', linewidth=1.5,
               label='Good threshold (0.7)')
    ax.set_xlabel('Split-Half Loading Correlation', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('EFA + Pupil: Factor Stability (200 splits)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    fig.savefig(OUTDIR / '08_stability_with_pupil.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 08_stability_with_pupil.png")


# ═════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("COMPARISON: With vs Without pupil_reward_peak")
    print("=" * 60)

    print("\n[1] Running EFA + pupil pipeline...")
    from pipeline_efa_with_pupil import run as efa_run
    efa_result = efa_run()

    print("\n[2] Decoder comparison bar chart...")
    plot_decoder_comparison()

    print("\n[3] EFA comparison bar chart...")
    plot_efa_comparison(efa_result)

    print("\n[4] EFA loadings heatmap (+pupil)...")
    plot_loadings(efa_result)

    print("\n[5] EFA factor scatter (+pupil)...")
    plot_factor_scatter(efa_result)

    print("\n[6] EFA group comparison (+pupil)...")
    plot_group_comparison(efa_result)

    print("\n[7] Decoder confusion matrices (+pupil)...")
    plot_confusion_matrices()

    print("\n[8] Summary table...")
    plot_summary_table(efa_result)

    print("\n[9] Stability histogram (+pupil)...")
    plot_stability_histogram(efa_result)

    print(f"\nAll figures saved to: {OUTDIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
