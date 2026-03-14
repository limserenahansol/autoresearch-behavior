"""
pipeline.py  --  Experiment #4: 2-class (morphine vs water) + NaN indicators
"""
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, '.')
from prepare import load_data, get_feature_matrix, evaluate_classification
from prepare import ALL_CANDIDATE_FEATURES


def run():
    df = load_data()

    X, mouse_keys, groups, periods, days, feat_names = \
        get_feature_matrix(df, ALL_CANDIDATE_FEATURES)

    is_active = (groups == 'Active').astype(float).reshape(-1, 1)
    X = np.column_stack([X, is_active])
    feat_names = feat_names + ['is_active']

    nan_indicators = np.isnan(X).astype(float)
    nan_cols_with_variance = np.std(nan_indicators, axis=0) > 0
    if nan_cols_with_variance.any():
        X = np.column_stack([X, nan_indicators[:, nan_cols_with_variance]])

    # 2-class target: morphine (During+Post+Re-exposure) vs water (Pre+Withdrawal)
    substance = np.array([
        'morphine' if p in ['During', 'Post', 'Re-exposure'] else 'water'
        for p in periods
    ])
    le = LabelEncoder()
    y = le.fit_transform(substance)

    mice = np.unique(mouse_keys)
    y_pred = np.full_like(y, -1)

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

        model = RandomForestClassifier(
            n_estimators=500, max_depth=12,
            min_samples_leaf=1, random_state=42,
        )
        model.fit(X_train, y[train_mask])
        y_pred[test_mask] = model.predict(X_test)

    result = evaluate_classification(y, y_pred, mouse_keys)

    print(f"METRIC per_mouse_acc={result['per_mouse_acc']:.6f}")
    print(f"METRIC accuracy={result['accuracy']:.6f}")
    print(f"METRIC f1_macro={result['f1_macro']:.6f}")
    print(f"N_FEATURES={X.shape[1]}")
    print(f"N_MICE={len(mice)}")
    print(f"TARGET_CLASSES={list(le.classes_)}")
    return result


if __name__ == "__main__":
    run()
