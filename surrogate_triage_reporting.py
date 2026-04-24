"""Helpers to map Holm correction onto per-column p-values with validity mask."""
from __future__ import annotations

import numpy as np

from surrogate_day_triage_core import holm_adjusted_pvalues


def holm_on_valid(pvals: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Holm adjustment on finite p-values where valid is True.
    Returns (holm_adjusted_p same shape as pvals with NaN invalid, reject_alpha005 bool array).
    """
    pvals = np.asarray(pvals, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    out_adj = np.full_like(pvals, np.nan, dtype=float)
    reject = np.zeros_like(valid, dtype=bool)
    idx = np.where(valid & np.isfinite(pvals))[0]
    if len(idx) == 0:
        return out_adj, reject
    adj = holm_adjusted_pvalues(pvals[idx])
    out_adj[idx] = adj
    reject[idx] = adj <= 0.05
    return out_adj, reject
