"""Predictor table, following Heldmyer et al. (2026) Sect. 2.2.

For each site and day t the predictors are:
  * daily weather and soil variables, raw;
  * 30-, 90- and 365-day rolling sums (precipitation, PET, P-PET) or means (others);
  * the variable-threshold (day-of-year) and fixed Weibull percentile of each of those;
  * monthly teleconnection indices (AMO, PDO, ENSO, PNA), untransformed;
  * decimal date (plus day-of-year sine/cosine, which the paper did not use).
P-PET percentiles stand in for SPEI: they rank the same climatic water balance
without fitting a log-logistic distribution.

Only data up to and including day t enter a row, so a row can be paired with a label
at t + lead to make a forecast (lead = 0 reproduces the paper's same-day nowcast).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .percentiles import doy_pools, fixed_percentile, variable_threshold_percentile

SUM_VARS = {"prcp", "pet", "wb"}  # wb = prcp - pet, the SPEI stand-in
WINDOWS = (30, 90, 365)


def _site_features(g: pd.DataFrame, ref_years: tuple[int, int] | None, percentiles: bool) -> pd.DataFrame:
    g = g.set_index("date").sort_index().asfreq("D")
    if {"prcp", "pet"} <= set(g.columns):
        g["wb"] = g["prcp"] - g["pet"]
    base_vars = [c for c in g.columns if c != "site"]
    feats = {}
    for v in base_vars:
        s = g[v].astype(float)
        feats[v] = s
        for w in WINDOWS:
            r = s.rolling(w, min_periods=int(w * 0.9))
            feats[f"{v}_{w}d"] = r.sum() if v in SUM_VARS else r.mean()
    f = pd.DataFrame(feats, index=g.index)
    if percentiles:
        mask = None
        if ref_years:
            mask = pd.Series((f.index.year >= ref_years[0]) & (f.index.year <= ref_years[1]), index=f.index)
        pools = doy_pools(f.index)
        pct = {}
        for c in list(f.columns):
            pct[f"{c}_vpct"] = variable_threshold_percentile(f[c], ref_mask=mask, pools=pools)
            pct[f"{c}_fpct"] = fixed_percentile(f[c], ref_mask=mask)
        f = pd.concat([f, pd.DataFrame(pct, index=f.index)], axis=1)
    f.index.name = "date"
    return f.reset_index()


def build_features(
    met: pd.DataFrame,
    teleconnections: pd.DataFrame | None = None,
    ref_years: tuple[int, int] | None = None,
    percentiles: bool = True,
    tele_lag_months: int = 1,
) -> pd.DataFrame:
    """met: long table (site, date, prcp, tmin, tmax, pet, sm_shallow, sm_deep, swe, ...).

    teleconnections: monthly table (date = first of month, one column per index). Each
    index is lagged by tele_lag_months because NOAA publishes them after the month ends.
    """
    parts = []
    for site, g in met.groupby("site"):
        f = _site_features(g, ref_years, percentiles)
        f.insert(0, "site", site)
        parts.append(f)
    X = pd.concat(parts, ignore_index=True)

    if teleconnections is not None:
        tc = teleconnections.copy()
        tc["date"] = pd.to_datetime(tc["date"]) + pd.DateOffset(months=tele_lag_months)
        tc = tc.set_index("date").sort_index()
        days = pd.date_range(X["date"].min(), X["date"].max(), freq="D")
        tc = tc.reindex(tc.index.union(days)).ffill().reindex(days)
        tc.index.name = "date"
        X = X.merge(tc.reset_index(), on="date", how="left")

    d = X["date"]
    X["decimal_date"] = d.dt.year + (d.dt.dayofyear - 1) / 365.25
    X["doy_sin"] = np.sin(2 * np.pi * d.dt.dayofyear / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * d.dt.dayofyear / 365.25)
    return X


def add_antecedent_flow(X: pd.DataFrame, flow_pct: pd.DataFrame) -> pd.DataFrame:
    """Optional, gauged sites only: today's flow percentile and its 7/30-day means.

    The paper left antecedent streamflow out so models could transfer to ungauged
    places. For a forecast at a gauge it is the strongest single predictor (drought
    persists), so it is offered as a switch.
    """
    parts = []
    for site, g in flow_pct.groupby("site"):
        s = g.set_index("date")["q_pct"].sort_index().asfreq("D")
        parts.append(pd.DataFrame({
            "site": site, "date": s.index, "qpct_0d": s.values,
            "qpct_7d": s.rolling(7, min_periods=5).mean().values,
            "qpct_30d": s.rolling(30, min_periods=25).mean().values,
        }))
    return X.merge(pd.concat(parts), on=["site", "date"], how="left")


def make_supervised(X: pd.DataFrame, labels: pd.DataFrame, lead_days: int) -> pd.DataFrame:
    """Pair predictors at day t with the drought label at day t + lead_days.

    labels: site, date, drought. Returns X plus columns target_date and y.
    """
    lab = labels[["site", "date", "drought"]].copy()
    lab["date"] = lab["date"] - pd.Timedelta(days=lead_days)
    lab = lab.rename(columns={"drought": "y"})
    out = X.merge(lab, on=["site", "date"], how="inner")
    out["target_date"] = out["date"] + pd.Timedelta(days=lead_days)
    return out.dropna(subset=["y"])
