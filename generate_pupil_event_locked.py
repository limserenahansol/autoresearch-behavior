"""
generate_pupil_event_locked.py  --  Event-Locked Pupil Trajectories
====================================================================
Loads frame-level data, extracts pupil traces locked to:
  1. Reward delivery (Injector_TTL onset)
  2. Lick bout onset (first lick after 2s pause)

Window: -2s to +5s around event, baseline = -2 to 0s mean.

Output:
  output/figures_pupil/08_pupil_reward_locked.png      - Reward-locked by period x group
  output/figures_pupil/09_pupil_lick_locked.png         - Lick-locked by period x group
  output/figures_pupil/10_pupil_reward_combined.png     - Reward-locked overlay
  output/figures_pupil/11_pupil_lick_combined.png       - Lick-locked overlay
  output/figures_pupil/12_pupil_event_delta.png         - Peak delta bar plot
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

RAW_CSV = Path(r"K:\addiction_concate_Dec_2025\longitudinal_outputs\run_009\ALL_mice_longitudinal.csv")
OUTPUT = Path(__file__).parent / "output" / "figures_pupil"
OUTPUT.mkdir(parents=True, exist_ok=True)

COL_ACTIVE = '#E53935'
COL_PASSIVE = '#1E88E5'

PERIOD_MAP = {}
for d in range(3, 6):   PERIOD_MAP[d] = 'Pre'
for d in range(6, 11):  PERIOD_MAP[d] = 'During'
for d in range(11, 14): PERIOD_MAP[d] = 'Post'
for d in range(14, 17): PERIOD_MAP[d] = 'Withdrawal'
for d in range(17, 19): PERIOD_MAP[d] = 'Re-exposure'

PERIOD_ORDER = ['Pre', 'During', 'Post', 'Withdrawal', 'Re-exposure']
PERIOD_COLORS = {
    'Pre': '#9E9E9E', 'During': '#42A5F5', 'Post': '#66BB6A',
    'Withdrawal': '#FFA726', 'Re-exposure': '#FF7043',
}

COHORT = {
    '6100_black': 'Active', '6100_red': 'Passive', '6100_orange': 'Passive',
    '0911_red': 'Active', '0911_orange': 'Passive', '0911_black': 'Passive', '0911_white': 'Active',
    '0910_red': 'Passive', '0910_orange': 'Passive', '0910_black': 'Active',
    '6099_red': 'Passive', '6099_orange': 'Active', '6099_black': 'Active', '6099_white': 'Passive',
}

PRE_WIN = 2.0
POST_WIN = 5.0
DT = 1.0 / 30
T_AXIS = np.arange(-PRE_WIN, POST_WIN + DT, DT)
BOUT_GAP = 2.0  # seconds gap to define new bout


def extract_traces(time, pupil, event_times, t_axis):
    """Extract pupil snippets around each event, baseline-subtract."""
    traces = []
    for evt in event_times:
        t_rel = time - evt
        mask = (t_rel >= t_axis[0] - DT) & (t_rel <= t_axis[-1] + DT)
        if mask.sum() < 10:
            continue
        snippet = np.interp(t_axis, t_rel[mask], pupil[mask])
        # Baseline subtract: mean of -2 to 0
        bl_mask = t_axis <= 0
        bl = np.nanmean(snippet[bl_mask])
        if np.isfinite(bl):
            snippet = snippet - bl
        traces.append(snippet)
    if len(traces) == 0:
        return None
    return np.array(traces)


def detect_events(time, ttl_signal):
    """Detect onset times from a TTL signal."""
    ttl = ttl_signal > 0.5
    edges = np.diff(np.concatenate(([False], ttl, [False])).astype(int))
    onsets = np.where(edges == 1)[0]
    onsets = onsets[onsets < len(time)]
    return time[onsets]


def detect_bout_onsets(time, lick_ttl):
    """Detect lick bout onsets (first lick after >=2s gap)."""
    lick_times = detect_events(time, lick_ttl)
    if len(lick_times) < 2:
        return lick_times
    ilis = np.diff(lick_times)
    bout_starts = [lick_times[0]]
    for i, ili in enumerate(ilis):
        if ili >= BOUT_GAP:
            bout_starts.append(lick_times[i + 1])
    return np.array(bout_starts)


def process_all_sessions():
    """Process raw CSV, extract event-locked pupil traces."""
    print("  Loading raw CSV and extracting event-locked traces...")

    cols = ['mouse_key', 'day_index', 'PupilTimestamp_s', 'Diameter_px',
            'Lick_TTL', 'Injector_TTL']

    reward_traces = []  # (mouse, day, period, group, mean_trace)
    lick_traces = []

    chunk_i = 0
    for chunk in pd.read_csv(RAW_CSV, usecols=cols, chunksize=300000):
        chunk_i += 1
        chunk = chunk.dropna(subset=['Diameter_px', 'PupilTimestamp_s'])
        chunk['mouse_key'] = chunk['mouse_key'].astype(str)
        chunk['group'] = chunk['mouse_key'].map(COHORT)
        chunk = chunk.dropna(subset=['group'])
        chunk['period'] = chunk['day_index'].map(PERIOD_MAP)
        chunk = chunk.dropna(subset=['period'])
        chunk['Lick_TTL'] = chunk['Lick_TTL'].fillna(0)
        chunk['Injector_TTL'] = chunk['Injector_TTL'].fillna(0)

        for (mk, day), ses in chunk.groupby(['mouse_key', 'day_index']):
            ses = ses.sort_values('PupilTimestamp_s')
            time = ses['PupilTimestamp_s'].values
            pupil = ses['Diameter_px'].values
            lick = ses['Lick_TTL'].values
            inj = ses['Injector_TTL'].values
            group = ses['group'].iloc[0]
            period = ses['period'].iloc[0]

            if len(time) < 100:
                continue

            # Reward-locked
            rew_times = detect_events(time, inj)
            if len(rew_times) >= 1:
                tr = extract_traces(time, pupil, rew_times, T_AXIS)
                if tr is not None:
                    reward_traces.append({
                        'mouse': mk, 'day': day, 'period': period,
                        'group': group, 'trace': np.nanmean(tr, axis=0),
                        'n_events': len(rew_times),
                    })

            # Lick bout onset-locked
            bout_onsets = detect_bout_onsets(time, lick)
            if len(bout_onsets) >= 3:
                tr = extract_traces(time, pupil, bout_onsets, T_AXIS)
                if tr is not None:
                    lick_traces.append({
                        'mouse': mk, 'day': day, 'period': period,
                        'group': group, 'trace': np.nanmean(tr, axis=0),
                        'n_events': len(bout_onsets),
                    })

        if chunk_i % 5 == 0:
            print(f"    Processed chunk {chunk_i} ({chunk_i * 300000:,} rows)...")

    print(f"  Reward-locked sessions: {len(reward_traces)}")
    print(f"  Lick bout-locked sessions: {len(lick_traces)}")

    return reward_traces, lick_traces


def plot_event_locked_separate(traces_list, t_axis, event_name, filename):
    """5 rows (periods) x 2 cols (Passive | Active)"""
    fig, axes = plt.subplots(5, 2, figsize=(18, 24), sharey=True, sharex=True)

    for row, period in enumerate(PERIOD_ORDER):
        period_traces = [t for t in traces_list if t['period'] == period]

        for col, (group, color) in enumerate([('Passive', COL_PASSIVE), ('Active', COL_ACTIVE)]):
            ax = axes[row, col]
            grp_traces = [t for t in period_traces if t['group'] == group]

            if len(grp_traces) == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', fontsize=12)
                ax.set_title(f'{period} - {group}', fontsize=12, fontweight='bold',
                             color=PERIOD_COLORS[period])
                continue

            # Per-mouse average (average across days)
            mice = list(set(t['mouse'] for t in grp_traces))
            mouse_avgs = []
            for m in mice:
                m_traces = [t['trace'] for t in grp_traces if t['mouse'] == m]
                mouse_avgs.append(np.nanmean(m_traces, axis=0))
                ax.plot(t_axis, mouse_avgs[-1], '-', color=color, alpha=0.25, linewidth=0.8)

            all_arr = np.array(mouse_avgs)
            mean_trace = np.nanmean(all_arr, axis=0)
            sem_trace = np.nanstd(all_arr, axis=0) / np.sqrt(len(mouse_avgs))

            ax.plot(t_axis, mean_trace, '-', color=color, linewidth=2.5, zorder=5)
            ax.fill_between(t_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                            color=color, alpha=0.15)

            ax.axvline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.6)
            ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
            ax.set_title(f'{period} - {group} (n={len(mice)} mice, {len(grp_traces)} sessions)',
                         fontsize=11, fontweight='bold', color=PERIOD_COLORS[period])
            ax.grid(alpha=0.15)
            ax.tick_params(labelsize=10)

            if col == 0:
                ax.set_ylabel('Pupil Change (px)', fontsize=10)
            if row == 4:
                ax.set_xlabel('Time from event (s)', fontsize=10)

    # Event marker annotation
    fig.text(0.5, 0.01, f'Vertical black line = {event_name} onset | Baseline = mean of -2 to 0s',
             ha='center', fontsize=11, fontstyle='italic', color='#555')

    fig.suptitle(f'Pupil Response Locked to {event_name}\n(baseline-subtracted, thin=individual mice, thick=mean+/-SEM)',
                 fontsize=16, fontweight='bold', y=1.01, color='#0D47A1')
    plt.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_event_locked_combined(traces_list, t_axis, event_name, filename):
    """5 panels (one per period), both groups overlaid"""
    fig, axes = plt.subplots(1, 5, figsize=(30, 5.5), sharey=True)

    for ax, period in zip(axes, PERIOD_ORDER):
        period_traces = [t for t in traces_list if t['period'] == period]

        for group, color, label in [('Active', COL_ACTIVE, 'Active'), ('Passive', COL_PASSIVE, 'Passive')]:
            grp_traces = [t for t in period_traces if t['group'] == group]
            if len(grp_traces) == 0:
                continue

            mice = list(set(t['mouse'] for t in grp_traces))
            mouse_avgs = []
            for m in mice:
                m_traces = [t['trace'] for t in grp_traces if t['mouse'] == m]
                mouse_avgs.append(np.nanmean(m_traces, axis=0))

            all_arr = np.array(mouse_avgs)
            mean_trace = np.nanmean(all_arr, axis=0)
            sem_trace = np.nanstd(all_arr, axis=0) / np.sqrt(len(mouse_avgs))

            ax.plot(t_axis, mean_trace, '-', color=color, linewidth=2.5,
                    label=f'{label} (n={len(mice)})')
            ax.fill_between(t_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                            color=color, alpha=0.15)

        ax.axvline(0, color='black', linewidth=1.5, alpha=0.6)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_title(period, fontsize=14, fontweight='bold', color=PERIOD_COLORS[period])
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel('Pupil Change from Baseline (px)', fontsize=11, fontweight='bold')

    fig.suptitle(f'Pupil Response Locked to {event_name}: Active vs Passive',
                 fontsize=16, fontweight='bold', y=1.04, color='#0D47A1')
    plt.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_peak_delta(reward_traces, lick_traces, t_axis):
    """Bar plot: peak pupil dilation (0-2s post-event) by period and group"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    post_mask = (t_axis >= 0.5) & (t_axis <= 3.0)

    for ax, traces_list, title in [(ax1, reward_traces, 'Reward Delivery'),
                                    (ax2, lick_traces, 'Lick Bout Onset')]:
        x = np.arange(len(PERIOD_ORDER))
        w = 0.35

        for i, (group, color, offset) in enumerate([
            ('Active', COL_ACTIVE, -w/2), ('Passive', COL_PASSIVE, w/2)
        ]):
            means = []
            sems = []
            for period in PERIOD_ORDER:
                grp_traces = [t for t in traces_list
                              if t['period'] == period and t['group'] == group]
                if len(grp_traces) == 0:
                    means.append(0)
                    sems.append(0)
                    continue

                mice = list(set(t['mouse'] for t in grp_traces))
                mouse_peaks = []
                for m in mice:
                    m_traces = [t['trace'] for t in grp_traces if t['mouse'] == m]
                    avg = np.nanmean(m_traces, axis=0)
                    peak = np.nanmean(avg[post_mask])
                    mouse_peaks.append(peak)

                means.append(np.mean(mouse_peaks))
                sems.append(np.std(mouse_peaks) / np.sqrt(len(mouse_peaks)) if len(mouse_peaks) > 1 else 0)

            ax.bar(x + offset, means, w, yerr=sems, color=color, alpha=0.7,
                   edgecolor='white', capsize=4, label=group)

            for j, (val, sem) in enumerate(zip(means, sems)):
                if val != 0:
                    ax.text(x[j] + offset, val + sem + 0.05, f'{val:.2f}',
                            ha='center', fontsize=8, fontweight='bold')

        # Stats
        for j, period in enumerate(PERIOD_ORDER):
            a_traces = [t for t in traces_list if t['period'] == period and t['group'] == 'Active']
            p_traces = [t for t in traces_list if t['period'] == period and t['group'] == 'Passive']
            if len(a_traces) >= 2 and len(p_traces) >= 2:
                a_mice = list(set(t['mouse'] for t in a_traces))
                p_mice = list(set(t['mouse'] for t in p_traces))
                a_peaks = [np.nanmean(np.nanmean([t['trace'] for t in a_traces if t['mouse'] == m], axis=0)[post_mask])
                           for m in a_mice]
                p_peaks = [np.nanmean(np.nanmean([t['trace'] for t in p_traces if t['mouse'] == m], axis=0)[post_mask])
                           for m in p_mice]
                try:
                    _, p = mannwhitneyu(a_peaks, p_peaks)
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    if sig:
                        y_max = max(max(a_peaks), max(p_peaks)) + 0.3
                        ax.text(j, y_max, sig, ha='center', fontsize=12, fontweight='bold')
                except Exception:
                    pass

        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(PERIOD_ORDER, fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Pupil Change 0.5-3s (px)', fontsize=11)
        ax.set_title(f'Peak Pupil Response: {title}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.15)
        ax.tick_params(labelsize=10)

    fig.suptitle('Pupil Dilation After Events: Active vs Passive by Period',
                 fontsize=15, fontweight='bold', y=1.02, color='#0D47A1')
    plt.tight_layout()
    fig.savefig(OUTPUT / '12_pupil_event_delta.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 12_pupil_event_delta.png")


def main():
    print("=" * 60)
    print("EVENT-LOCKED PUPIL TRAJECTORIES")
    print("=" * 60)

    reward_traces, lick_traces = process_all_sessions()

    print(f"\n[1] Reward-locked (separate panels)...")
    plot_event_locked_separate(reward_traces, T_AXIS, 'Reward Delivery',
                                '08_pupil_reward_locked.png')

    print(f"\n[2] Lick bout-locked (separate panels)...")
    plot_event_locked_separate(lick_traces, T_AXIS, 'Lick Bout Onset',
                                '09_pupil_lick_locked.png')

    print(f"\n[3] Reward-locked (combined)...")
    plot_event_locked_combined(reward_traces, T_AXIS, 'Reward Delivery',
                                '10_pupil_reward_combined.png')

    print(f"\n[4] Lick bout-locked (combined)...")
    plot_event_locked_combined(lick_traces, T_AXIS, 'Lick Bout Onset',
                                '11_pupil_lick_combined.png')

    print(f"\n[5] Peak delta bar plot...")
    plot_peak_delta(reward_traces, lick_traces, T_AXIS)

    print(f"\nAll figures in: {OUTPUT}")


if __name__ == "__main__":
    main()
