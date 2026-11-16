"""Weibull percentiles, as in Heldmyer et al. (2026) Eq. 1.

Variable-threshold percentiles rank each day against all values that fall within a
centered window of days-of-year (default 30 days) across every reference year, so a
"dry" July is judged against other Julys. Fixed percentiles rank against the whole record.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _weibull_rank(pool: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Weibull plotting position of x within pool: rank / (n + 1), ties at mid-rank."""
    pool = np.sort(pool[~np.isnan(pool)])
    n = pool.size
    out = np.full(x.shape, np.nan)
    if n == 0:
        return out
    ok = ~np.isnan(x)
    lo = np.searchsorted(pool, x[ok], side="left")
    hi = np.searchsorted(pool, x[ok], side="right")
    # mid-rank among ties; +1 because ranks start at 1
    rank = (lo + hi) / 2.0 + 0.5
    out[ok] = rank / (n + 1.0)
    return out


def doy_pools(index: pd.DatetimeIndex, window: int = 30) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each day of year: (mask of days in its window, mask of days that are that day).
    Compute once per site and reuse across variables; it is the slow part."""
    doy = index.dayofyear.values.copy()
    doy[doy == 366] = 365  # fold leap day onto Dec 31
    half = window // 2
    pools = []
    for d in range(1, 366):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, 365 - dist)  # circular
        pools.append((dist <= half, doy == d))
    return pools


def variable_threshold_percentile(
    s: pd.Series, window: int = 30, ref_mask: pd.Series | None = None, pools=None
) -> pd.Series:
    """Day-of-year percentile of a daily series.

    s: daily series with a DatetimeIndex (one site).
    window: centered window width in days of year.
    ref_mask: boolean series, True where a value may enter the reference pool
        (use training years only to avoid leaking the test period).
    pools: output of doy_pools for s.index, to skip recomputing it.
    """
    s = s.astype(float)
    pools = pools or doy_pools(s.index, window)
    vals = s.values
    ref = vals.copy()
    if ref_mask is not None:
        ref[~ref_mask.reindex(s.index, fill_value=False).values] = np.nan
    out = np.full(vals.shape, np.nan)
    for in_window, here in pools:
        out[here] = _weibull_rank(ref[in_window], vals[here])
    return pd.Series(out, index=s.index, name=s.name)


def fixed_percentile(s: pd.Series, ref_mask: pd.Series | None = None) -> pd.Series:
    vals = s.astype(float).values
    ref = vals.copy()
    if ref_mask is not None:
        ref[~ref_mask.reindex(s.index, fill_value=False).values] = np.nan
    return pd.Series(_weibull_rank(ref, vals), index=s.index, name=s.name)
