"""Metrics used in the paper (Cohen's kappa, sensitivity, specificity, balanced accuracy)
plus probabilistic scores (Brier, Brier skill vs climatology), which ctwater reports."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def brier(y, p) -> float:
    """Mean squared error of a probability forecast (written out so it works on any scikit-learn)."""
    return float(np.mean((np.clip(np.asarray(p, dtype=float), 0, 1) - np.asarray(y, dtype=float)) ** 2))


def scores(y, p, threshold: float = 0.5, p_clim=None) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    out = {
        "n": len(y), "drought_rate": y.mean(), "threshold": threshold,
        "kappa": cohen_kappa_score(y, yhat) if len(set(y)) > 1 or len(set(yhat)) > 1 else np.nan,
        "sensitivity": sens, "specificity": spec, "balanced_accuracy": np.nanmean([sens, spec]),
        "brier": brier(y, p),
    }
    if p_clim is not None:
        bc = brier(y, p_clim)
        out["brier_skill"] = 1 - out["brier"] / bc if bc > 0 else np.nan
    return out


def best_kappa_threshold(y, p, grid=np.arange(0.05, 0.96, 0.05)) -> float:
    """Threshold that maximizes kappa (the paper used this to pick 0.35 for donor models).
    Fit it on training or validation data, never on the test period."""
    y = np.asarray(y).astype(int)
    ks = [cohen_kappa_score(y, (np.asarray(p) >= t).astype(int)) for t in grid]
    return float(grid[int(np.nanargmax(ks))])
