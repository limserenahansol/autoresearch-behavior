"""
generate_pupil_timecourse.py  --  Within-Session Pupil Trajectory
==================================================================
Loads frame-level pupil data (~30fps), bins into 10s windows,
normalizes per session, and plots group-averaged trajectories.

Output:
  output/figures_pupil/05_pupil_timecourse_separate.png   - Passive | Active panels by period
  output/figures_pupil/06_pupil_timecourse_combined.png   - Overlay per period
  output/figures_pupil/07_pupil_timecourse_all_days.png   - All individual days
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, '.')

RAW_CSV = Path(r"K:\addiction_concate_Dec_2025\longitudinal_outputs\run_009\ALL_mice_longitudinal.csv")
OUTPUT = Path(__file__).parent / "output" / "figures_pupil"
OUTPUT.mkdir(parents=True, exist_ok=True)

COL_ACTIVE = '#E53935'
COL_PASSIVE = '#1E88E5'

PERIOD_MAP = {}
for d in range(3, 6):
    PERIOD_MAP[d] = 'Pre'
for d in range(6, 11):
    PERIOD_MAP[d] = 'During'
for d in range(11, 14):
    PERIOD_MAP[d] = 'Post'
for d in range(14, 17):
    PERIOD_MAP[d] = 'Withdrawal'
for d in range(17, 19):
    PERIOD_MAP[d] = 'Re-exposure'

PERIOD_ORDER = ['Pre', 'During', 'Post', 'Withdrawal', 'Re-exposure']
PERIOD_COLORS = {
    'Pre': '#9E9E9E',
    'During': '#42A5F5',
    'Post': '#66BB6A',
    'Withdrawal': '#FFA726',
    'Re-exposure': '#FF7043',
}

COHORT = {
    '6100_black': 'Active', '6100_red': 'Passive', '6100_orange': 'Passive',
    '0911_red': 'Active', '0911_orange': 'Passive', '0911_black': 'Passive', '0911_white': 'Active',
    '0910_red': 'Passive', '0910_orange': 'Passive', '0910_black': 'Active',
    '6099_red': 'Passive', '6099_orange': 'Active', '6099_black': 'Active', '6099_white': 'Passive',
}

BIN_SEC = 10


def load_and_bin():
    """Load raw CSV in chunks, bin pupil into 10s windows per session."""
    print("  Loading raw CSV in chunks (6.8M rows)...")

    cols = ['mouse_key', 'day_index', 'PupilTimestamp_s', 'Diameter_px']
    results = []

    for i, chunk in enumerate(pd.read_csv(RAW_CSV, usecols=cols, chunksize=500000)):
        chunk = chunk.dropna(subset=['Diameter_px', 'PupilTimestamp_s'])
        chunk = chunk[chunk['day_index'] >= 3]  # skip habituation

        chunk['mouse_key'] = chunk['mouse_key'].astype(str)
        chunk['group'] = chunk['mouse_key'].map(COHORT)
        chunk = chunk.dropna(subset=['group'])

        chunk['period'] = chunk['day_index'].map(PERIOD_MAP)
        chunk = chunk.dropna(subset=['period'])

        chunk['time_bin'] = (chunk['PupilTimestamp_s'] // BIN_SEC) * BIN_SEC

        binned = chunk.groupby(['mouse_key', 'group', 'day_index', 'period', 'time_bin']).agg(
            pupil_mean=('Diameter_px', 'mean'),
            pupil_std=('Diameter_px', 'std'),
            n_frames=('Diameter_px', 'count'),
        ).reset_index()

        results.append(binned)
        if (i + 1) % 3 == 0:
            print(f"    Processed {(i+1)*500000:,} rows...")

    df = pd.concat(results, ignore_index=True)
    print(f"  Binned data: {len(df):,} rows ({df['mouse_key'].nunique()} mice)")

    # Normalize per session: z-score within each mouse-day
    df['pupil_z'] = df.groupby(['mouse_key', 'day_index'])['pupil_mean'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    # Also normalize to first 30s baseline per session
    def baseline_norm(g):
        baseline = g[g['time_bin'] <= 30]['pupil_mean'].mean()
        if pd.isna(baseline) or baseline == 0:
            return g['pupil_mean'] * 0
        return (g['pupil_mean'] - baseline) / baseline * 100
    df['pupil_pct'] = df.groupby(['mouse_key', 'day_index'], group_keys=False).apply(
        lambda g: baseline_norm(g)
    ).reset_index(drop=True)

    return df


def plot_by_period_separate(df):
    """5 rows x 2 cols: each period, Passive vs Active"""
    fig, axes = plt.subplots(5, 2, figsize=(22, 28), sharey='row')

    max_time = 600  # 10 minutes

    for row, period in enumerate(PERIOD_ORDER):
        pdata = df[(df['period'] == period) & (df['time_bin'] <= max_time)]

        for col, (group, color, title) in enumerate([
            ('Passive', COL_PASSIVE, 'Passive'),
            ('Active', COL_ACTIVE, 'Active'),
        ]):
            ax = axes[row, col]
            gdata = pdata[pdata['group'] == group]

            if len(gdata) == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', fontsize=14)
                ax.set_title(f'{period} - {title}', fontsize=13, fontweight='bold',
                             color=PERIOD_COLORS[period])
                continue

            # Individual mice (thin lines)
            mice = gdata['mouse_key'].unique()
            for m in mice:
                # Average across days within this period for this mouse
                mdata = gdata[gdata['mouse_key'] == m]
                m_avg = mdata.groupby('time_bin')['pupil_z'].mean()
                ax.plot(m_avg.index / 60, m_avg.values, '-',
                        color=color, alpha=0.25, linewidth=0.8)

            # Group mean (thick)
            grp_mean = gdata.groupby('time_bin')['pupil_z'].mean()
            grp_sem = gdata.groupby('time_bin')['pupil_z'].sem()
            t_min = grp_mean.index / 60
            ax.plot(t_min, grp_mean.values, '-', color=color,
                    linewidth=2.5, zorder=5)
            ax.fill_between(t_min,
                            grp_mean.values - grp_sem.values,
                            grp_mean.values + grp_sem.values,
                            color=color, alpha=0.15)

            ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
            ax.set_title(f'{period} - {title} (n={len(mice)})',
                         fontsize=13, fontweight='bold', color=PERIOD_COLORS[period])
            ax.grid(alpha=0.15)
            ax.tick_params(labelsize=10)

            if col == 0:
                ax.set_ylabel('Pupil (z)', fontsize=11)
            if row == 4:
                ax.set_xlabel('Time in session (min)', fontsize=11)

    fig.suptitle('Within-Session Pupil Trajectory by Period\n(thin = individual mice, thick = mean +/- SEM)',
                 fontsize=18, fontweight='bold', y=1.01, color='#0D47A1')
    plt.tight_layout()
    fig.savefig(OUTPUT / '05_pupil_timecourse_separate.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_pupil_timecourse_separate.png")


def plot_by_period_combined(df):
    """5 panels (one per period), both groups overlaid"""
    fig, axes = plt.subplots(1, 5, figsize=(30, 6), sharey=True)

    max_time = 600

    for ax, period in zip(axes, PERIOD_ORDER):
        pdata = df[(df['period'] == period) & (df['time_bin'] <= max_time)]

        for group, color, label in [
            ('Active', COL_ACTIVE, 'Active'),
            ('Passive', COL_PASSIVE, 'Passive'),
        ]:
            gdata = pdata[pdata['group'] == group]
            if len(gdata) == 0:
                continue

            grp_mean = gdata.groupby('time_bin')['pupil_z'].mean()
            grp_sem = gdata.groupby('time_bin')['pupil_z'].sem()
            t_min = grp_mean.index / 60
            n_mice = gdata['mouse_key'].nunique()
            ax.plot(t_min, grp_mean.values, '-', color=color,
                    linewidth=2.5, label=f'{label} (n={n_mice})')
            ax.fill_between(t_min,
                            grp_mean.values - grp_sem.values,
                            grp_mean.values + grp_sem.values,
                            color=color, alpha=0.15)

        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_title(period, fontsize=14, fontweight='bold',
                     color=PERIOD_COLORS[period])
        ax.set_xlabel('Time (min)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel('Pupil (z-scored)', fontsize=12, fontweight='bold')

    fig.suptitle('Within-Session Pupil Trajectory: Active vs Passive by Period',
                 fontsize=16, fontweight='bold', y=1.03, color='#0D47A1')
    plt.tight_layout()
    fig.savefig(OUTPUT / '06_pupil_timecourse_combined.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 06_pupil_timecourse_combined.png")


def plot_all_days(df):
    """Spaghetti plot: each day as a separate line, colored by period"""
    fig, (ax_p, ax_a) = plt.subplots(1, 2, figsize=(22, 8), sharey=True)

    max_time = 600

    for ax, group, title in [(ax_p, 'Passive', 'Passive'), (ax_a, 'Active', 'Active')]:
        gdata = df[(df['group'] == group) & (df['time_bin'] <= max_time)]

        # Average across mice per day
        day_avg = gdata.groupby(['day_index', 'period', 'time_bin'])['pupil_z'].mean().reset_index()

        for _, (day, period) in day_avg[['day_index', 'period']].drop_duplicates().iterrows():
            ddata = day_avg[(day_avg['day_index'] == day)]
            color = PERIOD_COLORS[period]
            ax.plot(ddata['time_bin'] / 60, ddata['pupil_z'], '-',
                    color=color, alpha=0.5, linewidth=1.2)

        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Time in session (min)', fontsize=12, fontweight='bold')
        ax.set_title(f'{title} Group', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=11)

    ax_p.set_ylabel('Pupil (z-scored)', fontsize=12, fontweight='bold')

    legend_patches = [mpatches.Patch(color=PERIOD_COLORS[p], label=p) for p in PERIOD_ORDER]
    fig.legend(handles=legend_patches, loc='upper right', fontsize=10,
               bbox_to_anchor=(0.98, 0.95))

    fig.suptitle('Pupil Trajectory: All Days (colored by period)',
                 fontsize=16, fontweight='bold', y=1.02, color='#0D47A1')
    plt.tight_layout()
    fig.savefig(OUTPUT / '07_pupil_timecourse_all_days.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 07_pupil_timecourse_all_days.png")


def main():
    print("=" * 60)
    print("PUPIL WITHIN-SESSION TRAJECTORY")
    print("=" * 60)

    df = load_and_bin()

    # Save binned data for future use
    df.to_csv(OUTPUT.parent / 'pupil_binned_10s.csv', index=False)
    print(f"  Saved: pupil_binned_10s.csv ({len(df):,} rows)")

    print("\n[1] By period (Passive | Active)...")
    plot_by_period_separate(df)

    print("\n[2] By period (combined overlay)...")
    plot_by_period_combined(df)

    print("\n[3] All days spaghetti...")
    plot_all_days(df)

    print(f"\nAll figures in: {OUTPUT}")


if __name__ == "__main__":
    main()
