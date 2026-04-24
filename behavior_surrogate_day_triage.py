"""
Surrogate-day triage on behavior-only (mouse x day) data.

Runs:
  1) Population real-vs-surrogate decoding (LOMO) + label-shuffle null.
  2) Per-feature univariate decoding + permutation p-values + FDR.

Outputs under output/surrogate_day_triage_behavior/:
  population_real_fake.csv
  per_feature_triage.csv
  optional figure per_feature_real_fake.png

Usage:
  python behavior_surrogate_day_triage.py
  python behavior_surrogate_day_triage.py --csv path/to/features_day_level.csv --fast --n-shuffles 80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from surrogate_day_triage_core import (
    benjamini_hochberg,
    build_real_fake_dataset,
    default_binary_factory,
    fast_binary_factory,
    lomo_binary_scores,
    per_column_lomo_scan,
    permutation_p_value,
    shuffle_null_distribution,
)
from prepare import PERIOD_ORDER

OUTPUT = Path(__file__).parent / "output" / "surrogate_day_triage_behavior"


def load_behavior_frame(csv_path: str | None):
    if csv_path:
        df = pd.read_csv(csv_path)
        df["mouse_key"] = df["mouse_key"].astype(str)
        df["Group"] = df["Group"].astype(str)
        df["Period"] = pd.Categorical(df["Period"], categories=PERIOD_ORDER, ordered=True)
        df["day_index"] = df["day_index"].astype(int)
        return df
    from prepare import load_data

    return load_data()


def main():
    p = argparse.ArgumentParser(
        description="Behavior-only surrogate-day triage (real vs within-mouse random-day surrogate)."
    )
    p.add_argument("--csv", type=str, default=None, help="Override features CSV (else prepare.DATA_PATH).")
    p.add_argument(
        "--n-shuffles",
        type=int,
        default=80,
        help="Label-shuffle null draws for population; per-feature uses the same unless --fast (then capped).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast", action="store_true", help="Use LogisticRegression instead of RF for speed.")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    try:
        df = load_behavior_frame(args.csv)
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    from run_all_classifiers import build_features

    X, mouse_keys, groups, periods, days, feat_names = build_features(df)
    rng = np.random.default_rng(args.seed)
    X_aug, y_aug, mk_aug = build_real_fake_dataset(X, mouse_keys, rng)

    factory = fast_binary_factory if args.fast else default_binary_factory

    pop_acc, pop_bacc = lomo_binary_scores(X_aug, y_aug, mk_aug, factory)
    null_pop = shuffle_null_distribution(
        X_aug, y_aug, mk_aug, factory, args.n_shuffles, args.seed + 999
    )
    p_pop = permutation_p_value(pop_acc, null_pop)

    pop_df = pd.DataFrame(
        [
            {
                "lomo_accuracy": pop_acc,
                "lomo_balanced_accuracy": pop_bacc,
                "shuffle_median": float(np.median(null_pop)),
                "shuffle_q95": float(np.quantile(null_pop, 0.95)),
                "permutation_p": p_pop,
                "n_shuffles": args.n_shuffles,
                "n_rows_augmented": len(y_aug),
                "n_features": X.shape[1],
            }
        ]
    )
    pop_path = OUTPUT / "population_real_fake.csv"
    pop_df.to_csv(pop_path, index=False)
    print(f"Wrote {pop_path}")
    print(
        f"Population real-vs-surrogate LOMO acc={pop_acc:.4f}, balanced_acc={pop_bacc:.4f}, "
        f"shuffle p={p_pop:.4g} (median null {np.median(null_pop):.4f})"
    )

    n_sh_col = min(args.n_shuffles, 40) if args.fast else args.n_shuffles
    acc, pvals, bacc, valid = per_column_lomo_scan(
        X_aug,
        y_aug,
        mk_aug,
        factory,
        n_shuffles=n_sh_col,
        shuffle_seed=args.seed + 1000,
        scan_factory=fast_binary_factory,
    )
    fdr = benjamini_hochberg(np.where(valid, pvals, 1.0), q=0.05)
    rows = []
    for j, name in enumerate(feat_names):
        if not valid[j]:
            continue
        rows.append(
            {
                "feature": name,
                "lomo_accuracy": acc[j],
                "lomo_balanced_accuracy": bacc[j],
                "permutation_p": pvals[j],
                "fdr_q0.05": bool(fdr[j]),
            }
        )
    feat_df = pd.DataFrame(rows).sort_values("lomo_accuracy", ascending=False)
    feat_path = OUTPUT / "per_feature_triage.csv"
    feat_df.to_csv(feat_path, index=False)
    print(f"Wrote {feat_path} ({len(feat_df)} features tested)")

    if not args.no_plot and len(feat_df) > 0:
        try:
            import matplotlib.pyplot as plt

            top = feat_df.head(min(25, len(feat_df))).iloc[::-1]
            fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.22)))
            ax.barh(top["feature"], top["lomo_accuracy"], color="steelblue")
            ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
            ax.axvline(float(np.median(null_pop)), color="coral", linestyle=":", label="pop. shuffle median")
            ax.set_xlabel("LOMO accuracy (univariate real vs surrogate day)")
            ax.set_title("Surrogate-day triage: behavior features (top 25)")
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            fig_path = OUTPUT / "per_feature_real_fake.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"Wrote {fig_path}")
        except Exception as e:
            print(f"(skip plot) {e}")


if __name__ == "__main__":
    main()
