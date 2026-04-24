"""
Shared utilities for Michelle-style triage adapted to mouse-day behavior / neural data.

Real vs fake (behavior analogue, day-level data):
  - "Real" rows: observed (mouse, day) feature vectors with label y=1.
  - "Fake" rows: same mouse, feature vector copied from a *different* day of that mouse
    (random-day surrogate). Label y=0.  This mirrors "aligned to real event vs random time"
    when the only resolution is mouse x day rather than continuous time.

Evaluation:
  - Leave-one-mouse-out (LOMO) on the augmented dataset (each mouse contributes real+fake rows).
  - Trial / label shuffle null: permute y, rerun LOMO, build a null distribution of accuracy.
  - Per-column (per-feature or per-neuron) univariate decoding with the same protocol.

Multiple comparisons: Benjamini-Hochberg FDR on permutation p-values.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score


def build_real_fake_dataset(
    X: np.ndarray,
    mouse_keys: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack real rows (y=1) and within-mouse random-day surrogate rows (y=0).
    mouse_keys is duplicated so LOMO keeps all rows of a mouse in the same fold split.
    """
    n = X.shape[0]
    fake = np.empty_like(X)
    mice = np.unique(mouse_keys)
    for m in mice:
        idx = np.where(mouse_keys == m)[0]
        Xi = X[idx]
        if len(idx) == 1:
            others = np.setdiff1d(np.arange(n), idx, assume_unique=True)
            pick = rng.integers(0, len(others), size=1)[0]
            fake[idx[0]] = X[others[pick]]
            continue
        for k, i in enumerate(idx):
            choices = idx[idx != i]
            j = int(rng.choice(choices))
            fake[i] = X[j]

    X_aug = np.vstack([X, fake])
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    mk = np.concatenate([mouse_keys, mouse_keys])
    return X_aug, y, mk


def _impute_scale(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    col_medians = np.nanmedian(X_train, axis=0)
    X_tr = X_train.copy()
    X_te = X_test.copy()
    for j in range(X_tr.shape[1]):
        med = col_medians[j] if np.isfinite(col_medians[j]) else 0.0
        X_tr[np.isnan(X_tr[:, j]), j] = med
        X_te[np.isnan(X_te[:, j]), j] = med
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd
    return X_tr, X_te


def lomo_binary_scores(
    X: np.ndarray,
    y: np.ndarray,
    mouse_keys: np.ndarray,
    model_factory,
) -> tuple[float, float]:
    """
    Leave-one-mouse-out; returns (accuracy, balanced_accuracy) on concatenated predictions.
    """
    mice = np.unique(mouse_keys)
    y = np.asarray(y).astype(int)
    y_pred = np.full_like(y, -1)
    for m in mice:
        te = mouse_keys == m
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        X_tr, X_te = _impute_scale(X[tr], X[te])
        y_tr = y[tr]
        if len(np.unique(y_tr)) < 2:
            y_pred[te] = y_tr[0]
            continue
        clf = model_factory()
        clf.fit(X_tr, y_tr)
        y_pred[te] = clf.predict(X_te)
    valid = y_pred >= 0
    if valid.sum() == 0:
        return 0.0, 0.0
    acc = float(np.mean(y[valid] == y_pred[valid]))
    bacc = float(balanced_accuracy_score(y[valid], y_pred[valid]))
    return acc, bacc


def shuffle_null_distribution(
    X: np.ndarray,
    y: np.ndarray,
    mouse_keys: np.ndarray,
    model_factory,
    n_shuffles: int,
    seed: int,
) -> np.ndarray:
    """Permute labels globally, LOMO accuracy each shuffle."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_shuffles)
    y0 = np.asarray(y).astype(int).copy()
    for s in range(n_shuffles):
        perm = rng.permutation(len(y0))
        ys = y0[perm]
        acc, _ = lomo_binary_scores(X, ys, mouse_keys, model_factory)
        out[s] = acc
    return out


def permutation_p_value(obs: float, null_dist: np.ndarray) -> float:
    """Inclusive permutation p-value."""
    n = len(null_dist)
    ge = np.sum(null_dist >= obs)
    return float((1 + ge) / (1 + n))


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Returns boolean mask of rejections (discoveries)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros(m, dtype=bool)
    k = np.where(passed)[0].max()
    cutoff = ranked[k]
    return p <= cutoff


def per_column_lomo_scan(
    X: np.ndarray,
    y: np.ndarray,
    mouse_keys: np.ndarray,
    model_factory,
    n_shuffles: int,
    shuffle_seed: int,
    scan_factory=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each column j, univariate LOMO accuracy + shuffle null p-value.
    scan_factory: optional lighter classifier for per-column work (defaults to model_factory).
    Returns (acc_per_col, p_per_col, bacc_per_col, valid_mask).
    """
    mf = scan_factory if scan_factory is not None else model_factory
    n, d = X.shape
    acc = np.full(d, np.nan)
    bacc = np.full(d, np.nan)
    pval = np.full(d, np.nan)
    valid = np.zeros(d, dtype=bool)

    for j in range(d):
        Xj = X[:, j : j + 1]
        if not np.any(np.isfinite(Xj)):
            continue
        if np.nanstd(Xj) < 1e-12 and np.all(np.isfinite(Xj)):
            continue
        a, ba = lomo_binary_scores(Xj, y, mouse_keys, mf)
        null = shuffle_null_distribution(Xj, y, mouse_keys, mf, n_shuffles, shuffle_seed + j)
        acc[j] = a
        bacc[j] = ba
        pval[j] = permutation_p_value(a, null)
        valid[j] = True

    return acc, pval, bacc, valid


def default_binary_factory():
    return RandomForestClassifier(
        n_estimators=80,
        max_depth=6,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=1,
    )


def fast_binary_factory():
    """Lighter model for quick scans."""
    return LogisticRegression(
        max_iter=2500,
        class_weight="balanced",
        random_state=42,
    )
