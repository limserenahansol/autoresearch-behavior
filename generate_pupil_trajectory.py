"""
generate_pupil_trajectory.py  --  Pupil Diameter Trajectory Across Days
========================================================================
Plots pupil_mean trajectory per mouse, normalized to each mouse's
Pre-phase baseline, with group comparison (Active vs Passive).

Output:
  output/figures_pupil/01_pupil_trajectory_separate.png
  output/figures_pupil/02_pupil_trajectory_combined.png
  output/figures_pupil/03_pupil_delta_from_pre.png
  output/figures_pupil/04_pupil_group_stats.png
  output/pupil_trajectory.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu, kruskal

sys.path.insert(0, '.')
from prepare_efa import load_day_level

OUTPUT = Path(__file__).parent / "output" / "figures_pupil"
OUTPUT.mkdir(parents=True, exist_ok=True)

COL_ACTIVE = '#E53935'
COL_PASSIVE = '#1E88E5'

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


def load_pupil_data():
    df = load_day_level()
    df = df[['mouse_key', 'Group', 'Period', 'day_index', 'pupil_mean']].copy()
    df = df.dropna(subset=['pupil_mean'])

    # Compute per-mouse Pre baseline
    pre = df[df['Period'] == 'Pre'].groupby('mouse_key')['pupil_mean'].median()
    df['pupil_baseline'] = df['mouse_key'].map(pre)
    df['pupil_z'] = (df['pupil_mean'] - df['pupil_baseline']) / df.groupby('mouse_key')['pupil_mean'].transform('std')
    df['pupil_pct'] = (df['pupil_mean'] - df['pupil_baseline']) / df['pupil_baseline'] * 100
    df['pupil_delta'] = df['pupil_mean'] - df['pupil_baseline']

    return df


def plot_separate(df):
    """Two-panel: Passive (left) vs Active (right) — raw pupil with individual baselines"""
    fig, (ax_p, ax_a) = plt.subplots(1, 2, figsize=(22, 9), sharey=True)
    days = sorted(df['day_index'].unique())

    for ax, group, color, title in [
        (ax_p, 'Passive', COL_PASSIVE, 'Passive Group'),
        (ax_a, 'Active', COL_ACTIVE, 'Active Group'),
    ]:
        grp = df[df['Group'] == group]
        mice = sorted(grp['mouse_key'].unique())

        for m in mice:
            mdata = grp[grp['mouse_key'] == m].sort_values('day_index')
            ax.plot(mdata['day_index'], mdata['pupil_pct'], '-o',
                    color=color, alpha=0.35, linewidth=1, markersize=4)

        mean_day = grp.groupby('day_index')['pupil_pct'].mean()
        sem_day = grp.groupby('day_index')['pupil_pct'].sem()
        ax.plot(mean_day.index, mean_day.values, '-o', color=color,
                linewidth=3.5, markersize=8, zorder=5,
                markeredgecolor='black', markeredgewidth=0.8)
        ax.fill_between(mean_day.index,
                        mean_day.values - sem_day.values,
                        mean_day.values + sem_day.values,
                        color=color, alpha=0.15)

        for period, (d_start, d_end) in PERIOD_DAYS.items():
            ax.axvspan(d_start - 0.4, d_end + 0.4, alpha=0.06,
                       color=PERIOD_COLORS[period])

        ax.axhline(0, color='gray', linewidth=1, linestyle='--', label='Pre baseline')
        ax.set_xlabel('Day', fontsize=14, fontweight='bold')
        ax.set_title(f'Pupil Size Change ({title})',
                     fontsize=16, fontweight='bold', color=color, pad=12)
        ax.set_xticks(days)
        ax.grid(axis='y', alpha=0.15)
        ax.tick_params(labelsize=12)

        ax.text(0.02, 0.98, f'n = {len(mice)} mice\nthin = individual\nthick = mean +/- SEM',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    ax_p.set_ylabel('Pupil Size Change from Pre Baseline (%)', fontsize=13, fontweight='bold')

    # Period labels at bottom
    for ax in [ax_p, ax_a]:
        for period, (d_start, d_end) in PERIOD_DAYS.items():
            y_bot = ax.get_ylim()[0]
            ax.text((d_start + d_end) / 2, y_bot + (ax.get_ylim()[1] - y_bot) * 0.02,
                    period, ha='center', fontsize=9, fontweight='bold',
                    color=PERIOD_COLORS[period], fontstyle='italic')

    fig.suptitle('Pupil Diameter Trajectory Across Days\n(% change from Pre baseline)',
                 fontsize=20, fontweight='bold', y=1.02, color='#0D47A1')
    plt.tight_layout()
    fig.savefig(OUTPUT / '01_pupil_trajectory_separate.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_pupil_trajectory_separate.png")


def plot_combined(df):
    """Single panel overlay"""
    fig, ax = plt.subplots(figsize=(16, 8))
    days = sorted(df['day_index'].unique())

    for group, color, label in [
        ('Active', COL_ACTIVE, 'Active'),
        ('Passive', COL_PASSIVE, 'Passive'),
    ]:
        grp = df[df['Group'] == group]
        mice = sorted(grp['mouse_key'].unique())

        for m in mice:
            mdata = grp[grp['mouse_key'] == m].sort_values('day_index')
            ax.plot(mdata['day_index'], mdata['pupil_pct'], '-',
                    color=color, alpha=0.15, linewidth=0.8)

        mean_day = grp.groupby('day_index')['pupil_pct'].mean()
        sem_day = grp.groupby('day_index')['pupil_pct'].sem()
        ax.plot(mean_day.index, mean_day.values, '-o', color=color,
                linewidth=3, markersize=7, label=f'{label} (n={len(mice)})',
                markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        ax.fill_between(mean_day.index,
                        mean_day.values - sem_day.values,
                        mean_day.values + sem_day.values,
                        color=color, alpha=0.15)

    for period, (d_start, d_end) in PERIOD_DAYS.items():
        ax.axvspan(d_start - 0.4, d_end + 0.4, alpha=0.07,
                   color=PERIOD_COLORS[period])
        ax.text((d_start + d_end) / 2, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 30,
                period, ha='center', fontsize=11, fontweight='bold',
                color=PERIOD_COLORS[period])

    ax.axhline(0, color='gray', linewidth=1, linestyle='--')
    ax.set_xlabel('Day', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pupil Size Change from Pre Baseline (%)', fontsize=13, fontweight='bold')
    ax.set_title('Pupil Diameter Trajectory: Active vs Passive',
                 fontsize=18, fontweight='bold', pad=15, color='#0D47A1')
    ax.set_xticks(days)
    ax.legend(fontsize=13, loc='best')
    ax.grid(axis='y', alpha=0.15)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(OUTPUT / '02_pupil_trajectory_combined.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_pupil_trajectory_combined.png")


def plot_delta_by_period(df):
    """Bar plot: mean pupil change from Pre, by period and group"""
    periods = ['During', 'Post', 'Withdrawal', 'Re-exposure']

    # Per-mouse-per-period median
    mouse_period = df.groupby(['mouse_key', 'Group', 'Period'])['pupil_pct'].median().reset_index()

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(periods))
    w = 0.35

    for i, (group, color, offset) in enumerate([
        ('Active', COL_ACTIVE, -w/2),
        ('Passive', COL_PASSIVE, w/2),
    ]):
        means = []
        sems = []
        for period in periods:
            vals = mouse_period[(mouse_period['Group'] == group) &
                                (mouse_period['Period'] == period)]['pupil_pct']
            means.append(vals.mean() if len(vals) > 0 else 0)
            sems.append(vals.sem() if len(vals) > 1 else 0)

        bars = ax.bar(x + offset, means, w, yerr=sems, color=color, alpha=0.7,
                      edgecolor='white', capsize=4, error_kw={'linewidth': 1.5},
                      label=f'{group}')

        for j, (bar, val) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sems[j] + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # Stats per period
    for j, period in enumerate(periods):
        active_vals = mouse_period[(mouse_period['Group'] == 'Active') &
                                    (mouse_period['Period'] == period)]['pupil_pct']
        passive_vals = mouse_period[(mouse_period['Group'] == 'Passive') &
                                     (mouse_period['Period'] == period)]['pupil_pct']
        if len(active_vals) > 1 and len(passive_vals) > 1:
            try:
                _, p = mannwhitneyu(active_vals, passive_vals)
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'NS'
                y_max = max(active_vals.mean() + active_vals.sem(),
                            passive_vals.mean() + passive_vals.sem()) + 3
                ax.plot([j - w/2, j - w/2, j + w/2, j + w/2],
                        [y_max, y_max + 1, y_max + 1, y_max], color='black', linewidth=1)
                ax.text(j, y_max + 1.5, f'{sig}\np={p:.3f}', ha='center', fontsize=9)
            except Exception:
                pass

    ax.axhline(0, color='gray', linewidth=1, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=13, fontweight='bold')
    ax.set_ylabel('Pupil Size Change from Pre (%)', fontsize=13, fontweight='bold')
    ax.set_title('Pupil Diameter: Change from Pre Baseline by Period',
                 fontsize=16, fontweight='bold', pad=15, color='#0D47A1')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.15)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(OUTPUT / '03_pupil_delta_from_pre.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 03_pupil_delta_from_pre.png")


def plot_group_stats(df):
    """Period x Group box plots with individual dots"""
    periods = ['Pre', 'During', 'Post', 'Withdrawal', 'Re-exposure']
    mouse_period = df.groupby(['mouse_key', 'Group', 'Period'])['pupil_mean'].median().reset_index()

    fig, axes = plt.subplots(1, 5, figsize=(24, 6), sharey=True)

    for ax, period in zip(axes, periods):
        active = mouse_period[(mouse_period['Group'] == 'Active') &
                               (mouse_period['Period'] == period)]['pupil_mean']
        passive = mouse_period[(mouse_period['Group'] == 'Passive') &
                                (mouse_period['Period'] == period)]['pupil_mean']

        data = [active.values, passive.values] if len(active) > 0 and len(passive) > 0 else [[0], [0]]

        bp = ax.boxplot(data, tick_labels=['Active', 'Passive'],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(COL_ACTIVE)
        bp['boxes'][0].set_alpha(0.35)
        bp['boxes'][1].set_facecolor(COL_PASSIVE)
        bp['boxes'][1].set_alpha(0.35)

        rng = np.random.RandomState(42)
        if len(active) > 0:
            ax.scatter(1 + rng.normal(0, 0.04, len(active)), active,
                       c=COL_ACTIVE, s=50, zorder=5, edgecolors='black', linewidths=0.5)
        if len(passive) > 0:
            ax.scatter(2 + rng.normal(0, 0.04, len(passive)), passive,
                       c=COL_PASSIVE, s=50, zorder=5, edgecolors='black', linewidths=0.5)

        if len(active) > 1 and len(passive) > 1:
            try:
                _, p = mannwhitneyu(active, passive)
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'NS'
                y_max = max(active.max(), passive.max()) * 1.05
                ax.text(1.5, y_max, f'{sig}\np={p:.3f}', ha='center', fontsize=9, fontweight='bold')
            except Exception:
                pass

        ax.set_title(period, fontsize=14, fontweight='bold',
                     color=PERIOD_COLORS[period])
        ax.grid(axis='y', alpha=0.2)
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel('Pupil Diameter (raw)', fontsize=13, fontweight='bold')

    fig.suptitle('Pupil Diameter by Period: Active vs Passive (raw values)',
                 fontsize=16, fontweight='bold', y=1.02, color='#0D47A1')
    plt.tight_layout()
    fig.savefig(OUTPUT / '04_pupil_group_stats.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_pupil_group_stats.png")


def main():
    print("Loading pupil data...")
    df = load_pupil_data()
    print(f"  {len(df)} observations, {df['mouse_key'].nunique()} mice")
    print(f"  Pupil range: {df['pupil_mean'].min():.1f} - {df['pupil_mean'].max():.1f}")

    print("\n[1] Separate trajectory (Passive | Active)...")
    plot_separate(df)

    print("\n[2] Combined trajectory...")
    plot_combined(df)

    print("\n[3] Delta from Pre by period...")
    plot_delta_by_period(df)

    print("\n[4] Group stats per period...")
    plot_group_stats(df)

    # Save data
    out_df = df[['mouse_key', 'Group', 'Period', 'day_index',
                  'pupil_mean', 'pupil_baseline', 'pupil_pct', 'pupil_delta']].copy()
    out_df.to_csv(OUTPUT.parent / 'pupil_trajectory.csv', index=False)
    print(f"\n  Saved: pupil_trajectory.csv")
    print(f"\nAll figures in: {OUTPUT}")


if __name__ == "__main__":
    main()
