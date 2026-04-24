"""
Surrogate-day triage when neural (e.g. GRIN imaging) columns are available alongside behavior.

Expected neural CSV:
  - Columns: mouse_key, day_index, plus one column per unit (default prefix neuron_).

The script inner-joins behavior and neural on mouse_key + day_index, runs build_features on the
merged behavior columns, concatenates neural activity, then runs the same real-vs-surrogate LOMO +
shuffle-null protocol as behavior_surrogate_day_triage.py.

Outputs (output/surrogate_day_triage_neural_behavior/):
  population_real_fake.csv
  per_neuron_triage.csv
  per_behavior_feature_triage.csv

Dry run without a neural file:
  python neural_behavior_surrogate_day_triage.py --demo-synthetic --n-neurons 12

Usage:
  python neural_behavior_surrogate_day_triage.py --neural-csv path/to/neural_mouse_day.csv
  python neural_behavior_surrogate_day_triage.py --neural-csv neural.csv --csv behavior.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from behavior_surrogate_day_triage import load_behavior_frame
from surrogate_day_triage_core import (
    benjamini_hochberg,
    build_real_fake_dataset,
    default_binary_factory,
    fast_binary_factory,
    lomo_binary_scores,
    per_column_lomo_scan,
    permutation_p_value,
    shuffle_null_distribution,
    shuffle_null_summary,
)
from surrogate_triage_reporting import holm_on_valid

OUTPUT = Path(__file__).parent / "output" / "surrogate_day_triage_neural_behavior"


def _pick_neural_columns(neu: pd.DataFrame, neural_prefix: str) -> list[str]:
    cols = [c for c in neu.columns if c.startswith(neural_prefix)]
    if cols:
        return cols
    skip = {"mouse_key", "day_index", "Group", "Period", "mouse_id", "session"}
    return [c for c in neu.columns if c not in skip]


def main():
    p = argparse.ArgumentParser(
        description="Behavior + neural surrogate-day triage (real vs within-mouse random-day surrogate)."
    )
    p.add_argument("--csv", type=str, default=None, help="Behavior features CSV (default: prepare.DATA_PATH).")
    p.add_argument("--neural-csv", type=str, default=None, help="Neural mouse x day matrix CSV.")
    p.add_argument("--neural-prefix", type=str, default="neuron_", help="Column name prefix for neural units.")
    p.add_argument("--demo-synthetic", action="store_true", help="Add random neuron_* columns for a dry run.")
    p.add_argument("--n-neurons", type=int, default=10, help="With --demo-synthetic: number of synthetic neurons.")
    p.add_argument("--n-shuffles", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--dprime-min", type=float, default=1.0, help="Screening gate: d_prime_vs_shuffle >= this.")
    p.add_argument(
        "--screening-alpha",
        type=float,
        default=0.1,
        help="Screening gate: permutation_p_right_sided < this.",
    )
    args = p.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    if not args.neural_csv and not args.demo_synthetic:
        print(
            "Provide --neural-csv with columns [mouse_key, day_index, <neural columns>], "
            "or run with --demo-synthetic to generate placeholder neurons."
        )
        sys.exit(2)

    from run_all_classifiers import build_features

    rng = np.random.default_rng(args.seed)

    if args.demo_synthetic:
        df0 = load_behavior_frame(args.csv)
        Xb, mouse_keys, groups, periods, days, beh_names = build_features(df0)
        n, n_neu = Xb.shape[0], args.n_neurons
        noise = rng.standard_normal((n, n_neu)).astype(np.float64) * 0.15
        sig = (rng.standard_normal((n, 1)) @ rng.standard_normal((1, n_neu))) * 0.05
        Xn = noise + sig
        neu_names = [f"neuron_{k:03d}" for k in range(n_neu)]
        X = np.hstack([Xb, Xn])
        feat_names = list(beh_names) + neu_names
    else:
        df_beh = load_behavior_frame(args.csv)
        neu = pd.read_csv(args.neural_csv)
        neu["mouse_key"] = neu["mouse_key"].astype(str)
        neu["day_index"] = neu["day_index"].astype(int)
        neu_cols = _pick_neural_columns(neu, args.neural_prefix)
        if not neu_cols:
            raise SystemExit("No neural columns found; set --neural-prefix or add neuron_* columns.")
        sub = neu[["mouse_key", "day_index"] + neu_cols]
        merged = df_beh.merge(sub, on=["mouse_key", "day_index"], how="inner")
        if len(merged) < 10:
            raise SystemExit(f"Very few rows after merge ({len(merged)}); check keys and CSVs.")
        Xb, mouse_keys, groups, periods, days, beh_names = build_features(merged)
        Xn = merged[neu_cols].values.astype(float)
        if Xn.shape[0] != Xb.shape[0]:
            raise SystemExit("Neural and behavior row counts differ after merge.")
        neu_names = list(neu_cols)
        X = np.hstack([Xb, Xn])
        feat_names = list(beh_names) + neu_names

    X_aug, y_aug, mk_aug = build_real_fake_dataset(X, mouse_keys, rng)
    factory = fast_binary_factory if args.fast else default_binary_factory

    pop_acc, pop_bacc = lomo_binary_scores(X_aug, y_aug, mk_aug, factory)
    null_pop = shuffle_null_distribution(
        X_aug, y_aug, mk_aug, factory, args.n_shuffles, args.seed + 999
    )
    p_pop = permutation_p_value(pop_acc, null_pop)
    pop_sum = shuffle_null_summary(pop_acc, null_pop)
    pop_df = pd.DataFrame(
        [
            {
                "lomo_accuracy": pop_acc,
                "lomo_balanced_accuracy": pop_bacc,
                "shuffle_median": float(np.median(null_pop)),
                "shuffle_q95": float(np.quantile(null_pop, 0.95)),
                "shuffle_q99": float(np.quantile(null_pop, 0.99)),
                "shuffle_null_mean": pop_sum["shuffle_null_mean"],
                "shuffle_null_std": pop_sum["shuffle_null_std"],
                "d_prime_vs_shuffle": pop_sum["d_prime_vs_shuffle"],
                "exceeds_shuffle_q95": pop_sum["exceeds_shuffle_q95"],
                "exceeds_shuffle_q99": pop_sum["exceeds_shuffle_q99"],
                "permutation_p_right_sided": p_pop,
                "n_shuffles": args.n_shuffles,
                "n_rows_augmented": len(y_aug),
                "n_columns_total": X.shape[1],
                "demo_synthetic": bool(args.demo_synthetic),
                "screening_dprime_min": args.dprime_min,
                "screening_alpha": args.screening_alpha,
            }
        ]
    )
    pop_path = OUTPUT / "population_real_fake.csv"
    pop_df.to_csv(pop_path, index=False)
    print(f"Wrote {pop_path}")
    print(
        f"Population (behavior+neural) real-vs-surrogate LOMO acc={pop_acc:.4f}, "
        f"balanced_acc={pop_bacc:.4f}, p={p_pop:.4g}"
    )

    n_sh = min(args.n_shuffles, 40) if args.fast else args.n_shuffles
    acc, pvals, bacc, valid, null_mean, null_std, dprime, ex_q95 = per_column_lomo_scan(
        X_aug,
        y_aug,
        mk_aug,
        factory,
        n_shuffles=n_sh,
        shuffle_seed=args.seed + 2000,
        scan_factory=fast_binary_factory,
    )
    fdr = benjamini_hochberg(np.where(valid, pvals, 1.0), q=0.05)
    holm_adj, holm_reject = holm_on_valid(pvals, valid)

    neu_name_set = set(neu_names)
    beh_rows, neu_rows = [], []
    for j, name in enumerate(feat_names):
        if not valid[j]:
            continue
        gate = bool(dprime[j] >= args.dprime_min and pvals[j] < args.screening_alpha)
        row = {
            "name": name,
            "lomo_accuracy": acc[j],
            "lomo_balanced_accuracy": bacc[j],
            "shuffle_null_mean": null_mean[j],
            "shuffle_null_std": null_std[j],
            "d_prime_vs_shuffle": dprime[j],
            "exceeds_shuffle_q95": bool(ex_q95[j]),
            "permutation_p_right_sided": pvals[j],
            "holm_adjusted_p": holm_adj[j],
            "holm_pass_alpha0.05": bool(holm_reject[j]),
            "fdr_q0.05_bh": bool(fdr[j]),
            "gate_dprime_and_perm_p": gate,
        }
        if name in neu_name_set:
            neu_rows.append(row)
        else:
            beh_rows.append(row)

    if neu_rows:
        ndf = pd.DataFrame(neu_rows).sort_values("lomo_accuracy", ascending=False)
        ndf.to_csv(OUTPUT / "per_neuron_triage.csv", index=False)
        g = int(ndf["gate_dprime_and_perm_p"].sum())
        print(f"Wrote {OUTPUT / 'per_neuron_triage.csv'} ({len(ndf)} units, screening gate: {g})")
    if beh_rows:
        bdf = pd.DataFrame(beh_rows).sort_values("lomo_accuracy", ascending=False)
        bdf.to_csv(OUTPUT / "per_behavior_feature_triage.csv", index=False)
        g = int(bdf["gate_dprime_and_perm_p"].sum())
        print(f"Wrote {OUTPUT / 'per_behavior_feature_triage.csv'} ({len(bdf)} features, gate: {g})")


if __name__ == "__main__":
    main()
