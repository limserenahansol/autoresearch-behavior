"""
run_all_classifiers_with_pupil.py
=================================
Same best-config decoder (Stacking + NaN indicators + log-transform),
but with pupil_reward_peak merged from output/pupil_reward_peak.csv.

Saves outputs into  output/with_pupil/  so they don't overwrite originals.
"""
import sys
import csv
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_data, get_feature_matrix, evaluate_classification
from prepare import ALL_CANDIDATE_FEATURES

OUTPUT = Path(__file__).parent / "output" / "with_pupil"
OUTPUT.mkdir(parents=True, exist_ok=True)

PUPIL_PEAK_CSV = Path(__file__).parent / "output" / "pupil_reward_peak.csv"


def merge_pupil_peak(df):
    """Merge pupil_reward_peak into the day-level feature DataFrame."""
    pupil = pd.read_csv(PUPIL_PEAK_CSV)
    pupil['mouse_key'] = pupil['mouse_key'].astype(str)
    pupil['day_index'] = pupil['day_index'].astype(int)
    df = df.merge(pupil[['mouse_key', 'day_index', 'pupil_reward_peak']],
                  on=['mouse_key', 'day_index'], how='left')
    return df


def build_features(df):
    """Build the best feature set + pupil_reward_peak."""
    feature_cols = [c for c in ALL_CANDIDATE_FEATURES if c in df.columns]
    if 'pupil_reward_peak' not in feature_cols:
        feature_cols.append('pupil_reward_peak')

    X = df[feature_cols].values.astype(float)
    mouse_keys = df['mouse_key'].values
    groups = df['Group'].values
    periods = df['Period'].values.astype(str)
    days = df['day_index'].values
    feat_names = list(feature_cols)

    is_active = (groups == 'Active').astype(float).reshape(-1, 1)
    X = np.column_stack([X, is_active])
    feat_names.append('is_active')

    nan_indicators = np.isnan(X).astype(float)
    nan_with_var = np.std(nan_indicators, axis=0) > 0
    if nan_with_var.any():
        nan_names = [f'nan_{feat_names[i]}' for i in range(len(feat_names))
                     if i < len(nan_with_var) and nan_with_var[i]]
        X = np.column_stack([X, nan_indicators[:, nan_with_var]])
        feat_names = feat_names + nan_names

    skewed = ['RequirementLast', 'lick_freq_per_min', 'bout_n',
              'rew_n', 'rew_freq_per_min', 'Requirement_cum']
    for col in skewed:
        if col in feat_names:
            idx = feat_names.index(col)
            X = np.column_stack([X, np.log1p(np.abs(X[:, idx])).reshape(-1, 1)])
            feat_names = feat_names + [f'log_{col}']

    return X, mouse_keys, groups, periods, days, feat_names


def make_stacking_model():
    return StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
            ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
    )


def run_lomo_classifier(X, y, mouse_keys, model_factory):
    mice = np.unique(mouse_keys)
    y_pred = np.full_like(y, -1)
    n_classes = len(np.unique(y))
    y_proba = np.full((len(y), n_classes), np.nan)

    for mouse in mice:
        test_mask = mouse_keys == mouse
        train_mask = ~test_mask
        X_train, X_test = X[train_mask].copy(), X[test_mask].copy()

        col_medians = np.nanmedian(X_train, axis=0)
        for j in range(X_train.shape[1]):
            med = col_medians[j] if np.isfinite(col_medians[j]) else 0.0
            X_train[np.isnan(X_train[:, j]), j] = med
            X_test[np.isnan(X_test[:, j]), j] = med

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = model_factory()
        model.fit(X_train, y[train_mask])
        y_pred[test_mask] = model.predict(X_test)

        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X_test)
                y_proba[test_mask, :proba.shape[1]] = proba
            except Exception:
                pass

    return y_pred, y_proba


def main():
    print("=" * 70)
    print("DECODER + pupil_reward_peak  (same best Stacking config)")
    print("=" * 70)

    df = load_data()
    df = merge_pupil_peak(df)
    n_pupil = df['pupil_reward_peak'].notna().sum()
    print(f"  pupil_reward_peak merged: {n_pupil}/{len(df)} rows have data")

    X, mouse_keys, groups, periods, days, feat_names = build_features(df)
    print(f"  Total features: {X.shape[1]}")

    summary = []

    # ── Task 1: Period (5-class) ──────────────────────────────────────
    print("\n[1/3] Period classifier (5-class)...")
    le1 = LabelEncoder()
    le1.fit(['Pre', 'During', 'Post', 'Withdrawal', 'Re-exposure'])
    y1 = le1.transform(periods)

    t0 = time.time()
    y1_pred, y1_proba = run_lomo_classifier(X, y1, mouse_keys, make_stacking_model)
    elapsed1 = time.time() - t0

    r1 = evaluate_classification(y1, y1_pred, mouse_keys)
    pd.DataFrame({
        'mouse_key': mouse_keys, 'day_index': days, 'group': groups,
        'period': periods, 'y_true': le1.inverse_transform(y1),
        'y_pred': le1.inverse_transform(y1_pred),
        'correct': (y1 == y1_pred).astype(int),
    }).to_csv(OUTPUT / "predictions_5class.csv", index=False)
    np.save(OUTPUT / "proba_5class.npy", y1_proba)
    print(f"  per_mouse_acc={r1['per_mouse_acc']:.4f}  ({elapsed1:.1f}s)")
    summary.append(('Period (5-class)', r1, elapsed1))

    # ── Task 2: Substance (2-class) ──────────────────────────────────
    print("\n[2/3] Substance classifier...")
    substance = np.array([
        'morphine' if p in ['During', 'Post', 'Re-exposure'] else 'water'
        for p in periods
    ])
    le2 = LabelEncoder()
    y2 = le2.fit_transform(substance)

    t0 = time.time()
    y2_pred, y2_proba = run_lomo_classifier(X, y2, mouse_keys, make_stacking_model)
    elapsed2 = time.time() - t0

    r2 = evaluate_classification(y2, y2_pred, mouse_keys)
    pd.DataFrame({
        'mouse_key': mouse_keys, 'day_index': days, 'group': groups,
        'period': periods, 'y_true': le2.inverse_transform(y2),
        'y_pred': le2.inverse_transform(y2_pred),
        'correct': (y2 == y2_pred).astype(int),
    }).to_csv(OUTPUT / "predictions_2class.csv", index=False)
    np.save(OUTPUT / "proba_2class.npy", y2_proba)
    print(f"  per_mouse_acc={r2['per_mouse_acc']:.4f}  ({elapsed2:.1f}s)")
    summary.append(('Substance (2-class)', r2, elapsed2))

    # ── Task 3: Group (2-class) ──────────────────────────────────────
    print("\n[3/3] Group classifier...")
    is_active_idx = feat_names.index('is_active') if 'is_active' in feat_names else None
    drop_cols = set()
    if is_active_idx is not None:
        drop_cols.add(is_active_idx)
    nan_key = 'nan_is_active'
    if nan_key in feat_names:
        drop_cols.add(feat_names.index(nan_key))

    keep_cols = [i for i in range(X.shape[1]) if i not in drop_cols]
    X_group = X[:, keep_cols]

    le3 = LabelEncoder()
    y3 = le3.fit_transform(groups)

    t0 = time.time()
    y3_pred, y3_proba = run_lomo_classifier(X_group, y3, mouse_keys, make_stacking_model)
    elapsed3 = time.time() - t0

    r3 = evaluate_classification(y3, y3_pred, mouse_keys)
    pd.DataFrame({
        'mouse_key': mouse_keys, 'day_index': days, 'group': groups,
        'period': periods, 'y_true': le3.inverse_transform(y3),
        'y_pred': le3.inverse_transform(y3_pred),
        'correct': (y3 == y3_pred).astype(int),
    }).to_csv(OUTPUT / "predictions_group.csv", index=False)
    np.save(OUTPUT / "proba_group.npy", y3_proba)
    print(f"  per_mouse_acc={r3['per_mouse_acc']:.4f}  ({elapsed3:.1f}s)")
    summary.append(('Group (2-class)', r3, elapsed3))

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Task':<25} {'per_mouse_acc':>14} {'accuracy':>10} {'f1_macro':>10}")
    print("-" * 63)
    for name, r, _ in summary:
        print(f"{name:<25} {r['per_mouse_acc']:>14.4f} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f}")
    print("=" * 70)

    with open(OUTPUT / "summary_all_classifiers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "per_mouse_acc", "accuracy", "f1_macro", "duration_s"])
        for name, r, elapsed in summary:
            w.writerow([name, f"{r['per_mouse_acc']:.6f}", f"{r['accuracy']:.6f}",
                        f"{r['f1_macro']:.6f}", f"{elapsed:.1f}"])
    print(f"\nSummary saved: {OUTPUT / 'summary_all_classifiers.csv'}")


if __name__ == "__main__":
    main()
