"""
pipeline.py  --  The Experiment Pipeline (AI-modifiable)
========================================================
This is the ONLY file the AI agent modifies.

Current config (Experiment #0 = baseline):
  - Features : 29 behavioral/pupil/pharma + 1 group indicator = 30
  - Target   : 5-class period prediction (Pre/During/Post/Withdrawal/Re-exposure)
  - Model    : RandomForest (200 trees, max_depth=8)
  - CV       : Leave-One-Mouse-Out (14 folds)
  - Metric   : per_mouse_acc = 0.8884

Run:
    python pipeline.py
"""
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, '.')
from prepare import load_data, get_feature_matrix, evaluate_classification
from prepare import ALL_CANDIDATE_FEATURES


def run():
    """Main experiment function. Returns a dict of metrics."""

    # ── 1. Load data ────────────────────────────────────────────────
    df = load_data()

    # ── 2. Select features ──────────────────────────────────────────
    X, mouse_keys, groups, periods, days, feat_names = \
        get_feature_matrix(df, ALL_CANDIDATE_FEATURES)

    is_active = (groups == 'Active').astype(float).reshape(-1, 1)
    X = np.column_stack([X, is_active])
    feat_names = feat_names + ['is_active']

    # ── 3. Define target ────────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(periods)

    # ── 4. Leave-One-Mouse-Out cross-validation ─────────────────────
    mice = np.unique(mouse_keys)
    y_pred = np.full_like(y, -1)

    for mouse in mice:
        test_mask = mouse_keys == mouse
        train_mask = ~test_mask

        X_train, X_test = X[train_mask].copy(), X[test_mask].copy()

        # NaN imputation: train-set median
        col_medians = np.nanmedian(X_train, axis=0)
        for j in range(X_train.shape[1]):
            med = col_medians[j] if np.isfinite(col_medians[j]) else 0.0
            X_train[np.isnan(X_train[:, j]), j] = med
            X_test[np.isnan(X_test[:, j]), j] = med

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train & predict
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=2,
            random_state=42,
        )
        model.fit(X_train, y[train_mask])
        y_pred[test_mask] = model.predict(X_test)

    # ── 5. Evaluate ─────────────────────────────────────────────────
    result = evaluate_classification(y, y_pred, mouse_keys)

    print(f"METRIC per_mouse_acc={result['per_mouse_acc']:.6f}")
    print(f"METRIC accuracy={result['accuracy']:.6f}")
    print(f"METRIC f1_macro={result['f1_macro']:.6f}")
    print(f"N_FEATURES={len(feat_names)}")
    print(f"N_MICE={len(mice)}")
    print(f"TARGET_CLASSES={list(le.classes_)}")

    return result


if __name__ == "__main__":
    run()
