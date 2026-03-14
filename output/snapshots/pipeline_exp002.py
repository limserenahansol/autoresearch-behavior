"""
pipeline.py  --  Experiment #2: RF + feature interactions
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

    # Feature interactions: key behavioral ratios
    req_idx = feat_names.index('RequirementLast') if 'RequirementLast' in feat_names else None
    pup_idx = feat_names.index('pupil_mean') if 'pupil_mean' in feat_names else None
    lf_idx = feat_names.index('lick_freq_per_min') if 'lick_freq_per_min' in feat_names else None
    rf_idx = feat_names.index('rew_freq_per_min') if 'rew_freq_per_min' in feat_names else None

    extras = []
    extra_names = []
    if req_idx is not None and pup_idx is not None:
        extras.append(X[:, req_idx] * X[:, pup_idx])
        extra_names.append('req_x_pupil')
    if lf_idx is not None and rf_idx is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(X[:, rf_idx] > 0, X[:, lf_idx] / X[:, rf_idx], 0)
        extras.append(ratio)
        extra_names.append('lick_per_reward')
    if req_idx is not None:
        extras.append(np.log1p(np.abs(X[:, req_idx])))
        extra_names.append('log_req')

    if extras:
        X = np.column_stack([X] + [e.reshape(-1, 1) for e in extras])
        feat_names = feat_names + extra_names

    le = LabelEncoder()
    y = le.fit_transform(periods)

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
            n_estimators=200, max_depth=8,
            min_samples_leaf=2, random_state=42,
        )
        model.fit(X_train, y[train_mask])
        y_pred[test_mask] = model.predict(X_test)

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
