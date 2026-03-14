"""
visualize_group_by_period.py  --  Group decoder accuracy broken down by period
===============================================================================
Generates:
  output/figures/group_accuracy_by_period.png    - Bar chart per period
  output/figures/group_confusion_by_period.png   - Confusion matrices per period
  output/figures/group_biological_insight.png     - Pre vs Re-exposure comparison

Usage:
    python visualize_group_by_period.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

OUTPUT = Path(__file__).parent / "output"
FIGDIR = OUTPUT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

COL_ACTIVE = '#E53935'
COL_PASSIVE = '#1E88E5'
COL_OVERALL = '#4CAF50'

PERIODS = ['Pre', 'During', 'Post', 'Withdrawal', 'Re-exposure']


def load():
    return pd.read_csv(OUTPUT / "predictions_group.csv")


# ── Figure 1: Group accuracy by period (bar chart) ─────────────────

def plot_accuracy_by_period(df):
    fig, ax = plt.subplots(figsize=(12, 6))

    periods = PERIODS
    overall_accs = []
    active_accs = []
    passive_accs = []

    for p in periods:
        sub = df[df['period'] == p]
        overall_accs.append(sub['correct'].mean() if len(sub) > 0 else 0)
        a = sub[sub['group'] == 'Active']
        active_accs.append(a['correct'].mean() if len(a) > 0 else 0)
        p_sub = sub[sub['group'] == 'Passive']
        passive_accs.append(p_sub['correct'].mean() if len(p_sub) > 0 else 0)

    x = np.arange(len(periods))
    w = 0.25

    bars_a = ax.bar(x - w, active_accs, w, label='Active', color=COL_ACTIVE, edgecolor='white')
    bars_p = ax.bar(x, passive_accs, w, label='Passive', color=COL_PASSIVE, edgecolor='white')
    bars_o = ax.bar(x + w, overall_accs, w, label='Overall', color=COL_OVERALL, edgecolor='white')

    for bars, vals in [(bars_a, active_accs), (bars_p, passive_accs), (bars_o, overall_accs)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f'{val:.1%}', ha='center', fontsize=8, fontweight='bold')

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, label='Chance (50%)')
    ax.axhline(y=df['correct'].mean(), color='black', linestyle=':', alpha=0.4,
               label=f'All-period mean ({df["correct"].mean():.1%})')

    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Group Decoder (Active vs Passive): Accuracy by Experimental Period',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.2)

    # Annotation
    ax.annotate('No morphine experience yet\n(groups are indistinguishable)',
                xy=(0, overall_accs[0] + 0.05), fontsize=8, ha='center',
                color='gray', fontstyle='italic')
    ax.annotate('NaN pattern\ndrives separation',
                xy=(1, overall_accs[1] + 0.05), fontsize=8, ha='center',
                color='gray', fontstyle='italic')
    ax.annotate('Voluntary vs forced\nexperience separates groups',
                xy=(4, overall_accs[4] + 0.05), fontsize=8, ha='center',
                color='gray', fontstyle='italic')

    plt.tight_layout()
    fig.savefig(FIGDIR / 'group_accuracy_by_period.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/group_accuracy_by_period.png")


# ── Figure 2: Confusion matrices per period ─────────────────────────

def plot_confusion_by_period(df):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    for ax, period in zip(axes, PERIODS):
        sub = df[df['period'] == period]
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        cm = confusion_matrix(sub['y_true'], sub['y_pred'], labels=['Active', 'Passive'])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        disp = ConfusionMatrixDisplay(cm_pct, display_labels=['Active', 'Passive'])
        disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='.1f')

        acc = sub['correct'].mean()
        n = len(sub)
        ax.set_title(f'{period}\nacc={acc:.1%} (n={n})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Actual', fontsize=9)

    fig.suptitle('Group Decoder: Confusion Matrix by Period (% per row)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGDIR / 'group_confusion_by_period.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/group_confusion_by_period.png")


# ── Figure 3: Biological insight (Pre vs Re-exposure) ───────────────

def plot_biological_insight(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Pre vs Re-exposure accuracy comparison
    ax = axes[0]
    periods_compare = ['Pre', 'Re-exposure']
    accs = [df[df['period'] == p]['correct'].mean() for p in periods_compare]
    colors = ['#BDBDBD', '#FF7043']
    bars = ax.bar(periods_compare, accs, color=colors, edgecolor='white', width=0.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f'{val:.1%}', ha='center', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Group Classification Accuracy', fontsize=11)
    ax.set_title('A. Before vs After Morphine', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # Panel B: All periods trajectory
    ax = axes[1]
    period_accs = [df[df['period'] == p]['correct'].mean() for p in PERIODS]
    colors_all = ['#BDBDBD', '#42A5F5', '#66BB6A', '#FFA726', '#FF7043']
    bars = ax.bar(range(len(PERIODS)), period_accs, color=colors_all, edgecolor='white')
    ax.set_xticks(range(len(PERIODS)))
    ax.set_xticklabels(PERIODS, rotation=30, ha='right', fontsize=9)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    for i, (bar, val) in enumerate(zip(bars, period_accs)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f'{val:.1%}', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('B. Group Separability Across Phases', fontsize=12, fontweight='bold')

    # Panel C: Subset accuracy comparison
    ax = axes[2]
    subsets = {
        'All\nperiods': df['correct'].mean(),
        'W + R': df[df['period'].isin(['Withdrawal', 'Re-exposure'])]['correct'].mean(),
        'P + W + R': df[df['period'].isin(['Post', 'Withdrawal', 'Re-exposure'])]['correct'].mean(),
        'Re-exp\nonly': df[df['period'] == 'Re-exposure']['correct'].mean(),
    }
    names = list(subsets.keys())
    vals = list(subsets.values())
    colors_sub = ['#78909C', '#26A69A', '#29B6F6', '#FF7043']
    bars = ax.bar(names, vals, color=colors_sub, edgecolor='white', width=0.6)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f'{val:.1%}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('C. Accuracy by Data Subset', fontsize=12, fontweight='bold')

    fig.suptitle('Morphine Experience Creates Decodable Group Differences',
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig.savefig(FIGDIR / 'group_biological_insight.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/group_biological_insight.png")


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GENERATING GROUP-BY-PERIOD FIGURES")
    print("=" * 60)

    df = load()

    print("\n[1] Group accuracy by period...")
    plot_accuracy_by_period(df)

    print("\n[2] Confusion matrices by period...")
    plot_confusion_by_period(df)

    print("\n[3] Biological insight (Pre vs Re-exposure)...")
    plot_biological_insight(df)

    print(f"\nAll figures saved to: {FIGDIR}")


if __name__ == "__main__":
    main()
