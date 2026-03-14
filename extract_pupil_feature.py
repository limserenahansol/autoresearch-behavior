"""
extract_pupil_feature.py  --  Extract peak reward-locked pupil dilation
========================================================================
Reads the raw 6.8M row CSV, computes per-mouse-per-day:
  - pupil_reward_peak: mean baseline-subtracted dilation 0.5-3s after reward

Saves to: output/pupil_reward_peak.csv  (mergeable with features_day_level.csv)
"""
import numpy as np
import pandas as pd
from pathlib import Path

RAW_CSV = Path(r"K:\addiction_concate_Dec_2025\longitudinal_outputs\run_009\ALL_mice_longitudinal.csv")
OUTPUT = Path(__file__).parent / "output" / "pupil_reward_peak.csv"

PRE_WIN = 2.0
POST_WIN = 5.0
DT = 1.0 / 30
T_AXIS = np.arange(-PRE_WIN, POST_WIN + DT, DT)
PEAK_START = 0.5
PEAK_END = 3.0


def extract_traces(time, pupil, event_times, t_axis):
    traces = []
    for evt in event_times:
        t_rel = time - evt
        mask = (t_rel >= t_axis[0] - DT) & (t_rel <= t_axis[-1] + DT)
        if mask.sum() < 10:
            continue
        snippet = np.interp(t_axis, t_rel[mask], pupil[mask])
        bl_mask = t_axis <= 0
        bl = np.nanmean(snippet[bl_mask])
        if np.isfinite(bl):
            snippet = snippet - bl
        traces.append(snippet)
    return np.array(traces) if traces else None


def detect_events(time, ttl_signal):
    ttl = ttl_signal > 0.5
    edges = np.diff(np.concatenate(([False], ttl, [False])).astype(int))
    onsets = np.where(edges == 1)[0]
    onsets = onsets[onsets < len(time)]
    return time[onsets]


def main():
    print("Extracting peak reward-locked pupil dilation...")
    cols = ['mouse_key', 'day_index', 'PupilTimestamp_s', 'Diameter_px', 'Injector_TTL']
    peak_mask = (T_AXIS >= PEAK_START) & (T_AXIS <= PEAK_END)

    results = []
    chunk_i = 0
    for chunk in pd.read_csv(RAW_CSV, usecols=cols, chunksize=300000):
        chunk_i += 1
        chunk = chunk.dropna(subset=['Diameter_px', 'PupilTimestamp_s'])
        chunk['mouse_key'] = chunk['mouse_key'].astype(str)
        chunk['Injector_TTL'] = chunk['Injector_TTL'].fillna(0)

        for (mk, day), ses in chunk.groupby(['mouse_key', 'day_index']):
            ses = ses.sort_values('PupilTimestamp_s')
            time = ses['PupilTimestamp_s'].values
            pupil = ses['Diameter_px'].values
            inj = ses['Injector_TTL'].values

            if len(time) < 100:
                continue

            rew_times = detect_events(time, inj)
            if len(rew_times) < 1:
                continue

            tr = extract_traces(time, pupil, rew_times, T_AXIS)
            if tr is None:
                continue

            mean_trace = np.nanmean(tr, axis=0)
            peak_val = float(np.nanmean(mean_trace[peak_mask]))

            results.append({
                'mouse_key': mk,
                'day_index': int(day),
                'pupil_reward_peak': peak_val,
                'n_rewards': len(rew_times),
            })

        if chunk_i % 5 == 0:
            print(f"  Processed {chunk_i * 300000:,} rows...")

    df = pd.DataFrame(results)

    # Average across sessions if multiple per day (shouldn't happen but safe)
    df = df.groupby(['mouse_key', 'day_index']).agg(
        pupil_reward_peak=('pupil_reward_peak', 'mean'),
        n_rewards=('n_rewards', 'sum'),
    ).reset_index()

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT}")
    print(f"  {len(df)} mouse-day rows")
    print(f"  Mice: {df['mouse_key'].nunique()}")
    print(f"  Days: {sorted(df['day_index'].unique())}")
    print(f"  pupil_reward_peak range: {df['pupil_reward_peak'].min():.3f} to {df['pupil_reward_peak'].max():.3f}")


if __name__ == "__main__":
    main()
