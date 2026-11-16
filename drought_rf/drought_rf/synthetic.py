"""Fake Connecticut-like data with the real file layout, so the pipeline can be run and
tested before real data arrive. Nothing here is a result."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make(n_sites: int = 4, start: str = "1980-01-01", end: str = "2020-12-31", seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    doy = dates.dayofyear.values
    months = pd.date_range(start, end, freq="MS")
    # a slow climate oscillation that modulates rainfall, exposed as a fake "pdo"
    osc = np.sin(2 * np.pi * np.arange(len(months)) / 90) + rng.normal(0, 0.3, len(months))
    tele = pd.DataFrame({"date": months, "amo": rng.normal(0, 1, len(months)), "pdo": osc,
                         "nino34": rng.normal(0, 1, len(months)), "pna": rng.normal(0, 1, len(months))})
    osc_daily = pd.Series(osc, index=months).reindex(dates, method="ffill").values

    met, flow, static = [], [], []
    for i in range(n_sites):
        site = f"0118{i:04d}"
        area = rng.uniform(20, 500)
        storage = rng.uniform(60, 200)  # mm, bigger = slower response
        wet = rng.random(n) < 0.33 * (1 + 0.25 * osc_daily)
        prcp = np.where(wet, rng.gamma(0.8, 11, n), 0.0)
        tmean = 10 - 13 * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 3, n)
        tmin, tmax = tmean - 5, tmean + 5
        pet = np.clip(0.2 + 0.16 * tmean, 0, None)
        swe = np.zeros(n); sm = np.zeros(n); q = np.zeros(n)
        snow, bucket, gw = 0.0, storage / 2, 50.0
        for t in range(n):
            if tmean[t] < 0:
                snow += prcp[t]; rain = 0.0
            else:
                melt = min(snow, 3 * tmean[t]); snow -= melt; rain = prcp[t] + melt
            bucket += rain
            et = pet[t] * bucket / storage
            bucket -= et
            excess = max(0.0, bucket - storage); bucket = min(bucket, storage)
            gw += excess + 0.02 * bucket; bucket -= 0.02 * bucket
            base = 0.03 * gw; gw -= base
            swe[t], sm[t], q[t] = snow, bucket / storage, (base + 0.3 * excess) * area * 0.0409
        met.append(pd.DataFrame({"site": site, "date": dates, "prcp": prcp, "tmin": tmin, "tmax": tmax,
                                 "pet": pet, "sm_shallow": sm + rng.normal(0, 0.03, n),
                                 "sm_deep": pd.Series(sm).rolling(20, min_periods=1).mean().values, "swe": swe}))
        flow.append(pd.DataFrame({"site": site, "date": dates, "q": q * np.exp(rng.normal(0, 0.1, n))}))
        static.append({"site": site, "drain_area_km2": area, "storage_mm": storage,
                       "forest_pct": rng.uniform(30, 90), "elev_m": rng.uniform(10, 400)})
    return pd.concat(met, ignore_index=True), pd.concat(flow, ignore_index=True), tele, pd.DataFrame(static)
