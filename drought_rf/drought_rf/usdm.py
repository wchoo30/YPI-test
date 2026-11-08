"""Predict U.S. Drought Monitor levels (None, D0, D1, D2+) for Connecticut's planning regions.

This is the region-level companion to the gauge model in pipeline.py. It runs on data
that is already in the repo or freely downloadable from AWS:

  * label: the weekly Drought Monitor export in ../data/ct_drought_monitor_2000_2026.csv
    (percent of each planning region at each level or worse);
  * predictors: NOAA nClimGrid-Daily county weather (scripts/fetch_nclimgrid_counties.py).

A region's level in a week is the worst level that covers at least half its area. D3
covered half a region in only 67 region-weeks (2016-17) and D4 never did, so D2 and worse
are one class.

    python -m drought_rf.usdm --usdm ../data/ct_drought_monitor_2000_2026.csv \
        --weather data/raw/ct_county_daily.csv --out outputs/usdm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score

from .features import build_features

LEVELS = ["None", "D0", "D1", "D2+"]  # class 0..3

# Planning region -> NOAA county whose weather stands in for it. Counties and planning
# regions do not line up exactly (CT replaced counties with planning regions in 2022);
# each region takes the county that holds most of its area.
REGION_COUNTY = {
    "Capitol": "Hartford",
    "Greater Bridgeport": "Fairfield",
    "Lower Connecticut River Valley": "Middlesex",
    "Naugatuck Valley": "New Haven",
    "Northeastern Connecticut": "Windham",
    "Northwest Hills": "Litchfield",
    "South Central Connecticut": "New Haven",
    "Southeastern Connecticut": "New London",
    "Western Connecticut": "Fairfield",
}


def usdm_levels(usdm: pd.DataFrame, share: float = 50.0) -> pd.DataFrame:
    """Weekly level per region: worst Dk whose cumulative area share is >= `share` percent."""
    d = usdm.copy()
    d["region"] = d["County"].str.replace(" Planning Region", "", regex=False)
    d["date"] = pd.to_datetime(d["ValidStart"])
    lvl = np.zeros(len(d), dtype=int)
    for k, col in enumerate(["D0", "D1", "D2"], start=1):
        lvl = np.where(d[col] >= share, k, lvl)
    d["level"] = lvl
    d["d1_area"] = d["D1"]
    return d[["region", "date", "level", "d1_area"]].sort_values(["region", "date"]).reset_index(drop=True)


def hargreaves_pet(tmax, tmin, doy, lat_deg: float = 41.6):
    """Daily reference evapotranspiration (mm) from temperature alone (Hargreaves 1985)."""
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    tmean = (tmax + tmin) / 2
    return np.clip(0.0023 * 0.408 * ra * (tmean + 17.8) * np.sqrt(np.clip(tmax - tmin, 0, None)), 0, None)


def region_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """County daily weather -> one daily series per planning region (site = region)."""
    w = weather.copy()
    w["date"] = pd.to_datetime(w["date"])
    w = w[w["prcp"] >= 0]  # newest days can hold negative fill values
    w["county"] = w["region_name"].str.replace("CT: ", "", regex=False).str.replace(" County", "", regex=False)
    w["pet"] = hargreaves_pet(w["tmax"].values, w["tmin"].values, w["date"].dt.dayofyear.values)
    parts = []
    for region, county in REGION_COUNTY.items():
        g = w[w["county"] == county][["date", "prcp", "tmax", "tmin", "pet"]].copy()
        g.insert(0, "site", region)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def make_table(levels: pd.DataFrame, X: pd.DataFrame, lead_weeks: int) -> pd.DataFrame:
    """Rows = region-weeks. Predictors known on the USDM week's start date t, plus the
    region's current level and its change over the last 4 weeks; label = level at t + lead."""
    lv = levels.rename(columns={"region": "site"})
    cur = lv[["site", "date", "level", "d1_area"]].rename(columns={"level": "level_now", "d1_area": "d1_area_now"})
    cur = cur.sort_values(["site", "date"])
    cur["level_4wk_ago"] = cur.groupby("site")["level_now"].shift(4)
    cur["d1_area_change_4wk"] = cur["d1_area_now"] - cur.groupby("site")["d1_area_now"].shift(4)
    tgt = lv[["site", "date", "level"]].copy()
    tgt["date"] = tgt["date"] - pd.Timedelta(weeks=lead_weeks)
    t = cur.merge(tgt.rename(columns={"level": "y"}), on=["site", "date"], how="inner")
    t = t.merge(X, on=["site", "date"], how="inner")
    t["target_date"] = t["date"] + pd.Timedelta(weeks=lead_weeks)
    return t


def _rf(seed=0):
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=10, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=seed)


def _scores(y, pred, prob_d1):
    y, pred = np.asarray(y), np.asarray(pred)
    yb, pb = (y >= 2).astype(int), (pred >= 2).astype(int)
    tp = int(((yb == 1) & (pb == 1)).sum()); fn = int(((yb == 1) & (pb == 0)).sum())
    tn = int(((yb == 0) & (pb == 0)).sum()); fp = int(((yb == 0) & (pb == 1)).sum())
    return {
        "n": len(y),
        "exact_level": float((y == pred).mean()),
        "within_one_level": float((np.abs(y - pred) <= 1).mean()),
        "kappa_weighted": float(cohen_kappa_score(y, pred, weights="quadratic")) if len(set(y)) > 1 else np.nan,
        "d1plus_hit_rate": tp / (tp + fn) if tp + fn else np.nan,
        "d1plus_false_alarm": fp / (fp + tp) if fp + tp else np.nan,
        "d1plus_brier": float(np.mean((np.asarray(prob_d1) - yb) ** 2)),
    }


MODELS = ["climatology", "persistence", "rf_weather", "rf_weather_plus_now"]


def summarize(P: pd.DataFrame, alarm: float = 0.3) -> pd.DataFrame:
    """Scores per lead and model. Besides overall accuracy, scores *onsets*: weeks where a
    region is below D1 now and at D1 or worse at the target week. Persistence can never
    catch an onset, so this is where a forecast earns its keep. For the forests an onset
    alarm is P(D1 or worse) >= `alarm`; for the baselines it is their predicted level."""
    rows = []
    for lead, g in P.groupby("lead_weeks"):
        on = g[(g["level_now"] < 2) & (g["y"] >= 2)]
        quiet = g[(g["level_now"] < 2) & (g["y"] < 2)]
        for name in MODELS:
            pcol = name + "_p_d1plus"
            pd1 = g[pcol] if pcol in g else (g[name] >= 2).astype(float)
            alarm_on = (on[pcol] >= alarm) if pcol in g else (on[name] >= 2)
            alarm_quiet = (quiet[pcol] >= alarm) if pcol in g else (quiet[name] >= 2)
            rows.append({"lead_weeks": lead, "model": name, **_scores(g["y"], g[name], pd1),
                         "onsets": len(on),
                         "onset_caught": float(alarm_on.mean()) if len(on) else np.nan,
                         "onset_false_alarm_rate": float(alarm_quiet.mean()) if len(quiet) else np.nan})
    return pd.DataFrame(rows)


def run(usdm_csv, weather_csv, out_dir="outputs/usdm", leads=(0, 1, 2, 4),
        first_test_year=2010, last_test_year=2025, embargo_weeks=4, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    levels = usdm_levels(pd.read_csv(usdm_csv, dtype={"FIPS": str}))
    met = region_weather(pd.read_csv(weather_csv))
    # percentiles from 1991-2020, the climate normal period; fixed before any test year
    X = build_features(met, None, ref_years=(1991, 2020))
    log(f"{levels['region'].nunique()} regions, {len(levels)} region-weeks, weather to {met['date'].max().date()}")

    preds, imps = [], []
    for lead in leads:
        T = make_table(levels, X, lead)
        weather_cols = [c for c in X.columns if c not in ("site", "date")]
        sets = {"rf_weather": weather_cols,
                "rf_weather_plus_now": weather_cols + ["level_now", "level_4wk_ago", "d1_area_now", "d1_area_change_4wk"]}
        for yr in range(first_test_year, last_test_year + 1):
            test = T["date"].dt.year == yr
            train = T["target_date"] < pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(weeks=embargo_weeks)
            if not test.any() or train.sum() < 200:
                continue
            Tr, Te = T[train], T[test]
            out = Te[["site", "date", "target_date", "y", "level_now"]].copy()
            out["lead_weeks"], out["test_year"] = lead, yr
            out["persistence"] = Te["level_now"].values
            # climatology: most common level in training for that region and month
            clim = Tr.groupby([Tr["site"], Tr["target_date"].dt.month])["y"].agg(lambda s: s.mode().iloc[0])
            out["climatology"] = [clim.get((s, m), 0) for s, m in zip(Te["site"], Te["target_date"].dt.month)]
            for name, cols in sets.items():
                m = _rf().fit(Tr[cols].fillna(Tr[cols].median()), Tr["y"])
                P = m.predict_proba(Te[cols].fillna(Tr[cols].median()))
                classes = list(m.classes_)
                full = np.zeros((len(Te), 4))
                for j, c in enumerate(classes):
                    full[:, c] = P[:, j]
                out[name] = full.argmax(1)
                out[name + "_p_d1plus"] = full[:, 2:].sum(1)
                for k, lvl in enumerate(LEVELS):
                    out[f"{name}_p_{lvl}"] = full[:, k]
                if name == "rf_weather_plus_now":
                    imps.append(pd.Series(m.feature_importances_, index=cols, name=f"lead{lead}_{yr}"))
            preds.append(out)
        log(f"lead {lead} wk done")

    P = pd.concat(preds, ignore_index=True)
    S = summarize(P)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / "predictions.csv", index=False)
    S.to_csv(out / "summary.csv", index=False)
    if imps:
        pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False).rename("importance").to_csv(out / "rf_importance.csv")
    log(S.round(3).to_string(index=False))
    return P, S


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m drought_rf.usdm", description=__doc__.split("\n\n")[0])
    ap.add_argument("--usdm", required=True, help="Drought Monitor export CSV (repo data/ folder)")
    ap.add_argument("--weather", required=True, help="nClimGrid county CSV from scripts/fetch_nclimgrid_counties.py")
    ap.add_argument("--out", default="outputs/usdm")
    ap.add_argument("--leads", default="0,1,2,4", help="weeks ahead, comma separated")
    ap.add_argument("--first-test-year", type=int, default=2010)
    ap.add_argument("--last-test-year", type=int, default=2025)
    a = ap.parse_args()
    run(a.usdm, a.weather, a.out, tuple(int(x) for x in a.leads.split(",")), a.first_test_year, a.last_test_year)
