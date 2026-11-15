"""Live Drought Monitor forecasts for Connecticut's planning regions, and a running test.

Each week this
  1. downloads the newest data: the Drought Monitor's county statistics for CT (its web
     service) and NOAA nClimGrid-Daily county weather (AWS, about 4 days behind real time);
  2. trains the forest on every week so far and forecasts each region's level 1, 2 and 4
     weeks after the latest Drought Monitor map;
  3. adds the forecast to forecasts/log.csv, never overwriting an earlier one, so the log
     is an honest record of what the model said before the answer was known;
  4. scores every logged forecast whose target week has since been mapped.

    python -m drought_rf.live all          # update + forecast + verify
    python -m drought_rf.live update       # only download
    python -m drought_rf.live forecast     # only forecast (uses the downloaded files)
    python -m drought_rf.live verify       # only score the log

Files: data/raw/usdm_ct_live.csv, data/raw/ct_county_daily_live.csv (downloads, not in git);
forecasts/latest.csv, forecasts/log.csv, forecasts/verification.csv (small, meant for git).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_features
from .scorecard import binary_metrics, level_metrics
from .usdm import LEVELS, _rf, current_state, make_table, region_weather, usdm_levels

NCG = "https://noaa-nclimgrid-daily-pds.s3.amazonaws.com"
USDM_API = ("https://usdmdataservices.unl.edu/api/CountyStatistics/"
            "GetDroughtSeverityStatisticsByAreaPercent?aoi=CT&startdate={start}&enddate={end}&statisticsType=1")
UA = {"User-Agent": "drought_rf/0.1 (research prototype)"}
FEATURES_NOW = ["level_now", "level_4wk_ago", "d1_area_now", "d1_area_change_4wk"]
ALARM = 0.3  # P(D1 or worse) that counts as a new-drought warning, same as the backtest


def _get(url, accept=None, tries=4):
    h = dict(UA, **({"Accept": accept} if accept else {}))
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120).read()
        except Exception:
            if i == tries - 1:
                raise
    return b""

# ----------------------------------------------------------------------------- downloads


def fetch_usdm(out_csv, fallback_csv=None, start="1/1/2000") -> pd.DataFrame:
    """All CT county/planning-region statistics since 2000 from the Drought Monitor service.
    If the service can't be reached, uses fallback_csv (the repo's export)."""
    t = datetime.now()
    end = f"{t.month}/{t.day}/{t.year}"
    try:
        raw = _get(USDM_API.format(start=start, end=end), accept="text/csv")
        d = pd.read_csv(io.BytesIO(raw), dtype={"FIPS": str})
        if "County" not in d or "ValidStart" not in d:  # JSON came back, or columns renamed
            d = pd.read_json(io.BytesIO(_get(USDM_API.format(start=start, end=end), accept="application/json")))
            want = ["MapDate", "FIPS", "County", "State", "None", "D0", "D1", "D2", "D3", "D4", "ValidStart", "ValidEnd"]
            d = d.rename(columns={c: w for c in d.columns for w in want if c.lower() == w.lower()})
        d = d[d["County"].astype(str).str.contains("Planning Region")]
        if d.empty:
            raise ValueError("no planning regions in the response")
        print(f"Drought Monitor: {d['County'].nunique()} regions, maps to {pd.to_datetime(d['ValidStart']).max().date()}")
    except Exception as e:
        if not fallback_csv:
            raise
        print(f"Drought Monitor service not reachable ({type(e).__name__}); using {fallback_csv}")
        d = pd.read_csv(fallback_csv, dtype={"FIPS": str})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_csv, index=False)
    return d


def _epinoaa_keys(year):
    t = _get(f"{NCG}/?list-type=2&prefix=EpiNOAA/v1-0-0/parquet/cty/YEAR={year}/").decode()
    by = {}
    for k in re.findall(r"<Key>(.*?)</Key>", t):
        m = k.rsplit("/", 1)[1][:6]
        if m not in by or "scaled" in k:
            by[m] = k
    return list(by.values())


def _epinoaa(k):
    d = pd.read_parquet(io.BytesIO(_get(f"{NCG}/{k}")))
    d = d[d["region_name"].astype(str).str.startswith("CT:")]
    d = d[["region_name", "date", "tmax", "tmin", "prcp"]].copy()
    for c in ("tmax", "tmin", "prcp"):  # some months store numbers as text
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def _monthly_keys(year):
    t = _get(f"{NCG}/?list-type=2&prefix=access/averages/{year}/").decode()
    by = {}
    for k in re.findall(r"<Key>(access/averages/\d{4}/(prcp|tmax|tmin)-(\d{6})-cty-(scaled|prelim)\.csv)</Key>", t):
        key, var, ym, kind = k
        if (var, ym) not in by or kind == "scaled":
            by[(var, ym)] = key
    return by


def _monthly(key):
    """One access/averages county file (one row per county, one column per day) -> long table."""
    var, ym = re.search(r"/(prcp|tmax|tmin)-(\d{6})-", key).groups()
    rows = [r for r in _get(f"{NCG}/{key}").decode().splitlines() if ",CT: " in r]
    out = []
    for r in rows:
        f = [x.strip() for x in r.split(",")]
        name, vals = f[2], [float(x) for x in f[6:]]
        for day, v in enumerate(vals, start=1):
            if v > -999:
                out.append((name, f"{ym[:4]}-{ym[4:]}-{day:02d}", var, v))
    return pd.DataFrame(out, columns=["region_name", "date", "var", "value"])


def fetch_weather(out_csv, cache_dir="data/raw/_nclimgrid_cache", history_start=1990) -> pd.DataFrame:
    """nClimGrid-Daily county weather for CT, history_start to ~4 days ago.

    History comes from the EpiNOAA parquet copy (fast, cached after the first run); months it
    does not have yet come from NOAA's monthly county files, re-downloaded on every run because
    the newest month keeps filling in."""
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    hist_f = cache / "epinoaa_ct.csv"
    if hist_f.exists():
        hist = pd.read_csv(hist_f, parse_dates=["date"])
    else:
        print("First run: downloading weather history (about 1 minute)...", flush=True)
        keys = [k for y in range(history_start, datetime.now().year + 1) for k in _epinoaa_keys(y)]
        with cf.ThreadPoolExecutor(16) as ex:
            hist = pd.concat(list(ex.map(_epinoaa, keys)), ignore_index=True)
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist[hist["prcp"] >= 0].sort_values(["region_name", "date"])
        hist.to_csv(hist_f, index=False)
    # months after the history's last complete month
    last_full = (hist["date"].max() + pd.Timedelta(days=1)).to_period("M") - 1
    need = pd.period_range(last_full + 1, pd.Timestamp.today().to_period("M"), freq="M")
    keys = []
    for y in sorted({p.year for p in need}):
        keys += [k for (v, ym), k in _monthly_keys(y).items() if pd.Period(f"{ym[:4]}-{ym[4:]}", "M") in need]
    if keys:
        with cf.ThreadPoolExecutor(16) as ex:
            new = pd.concat(list(ex.map(_monthly, keys)), ignore_index=True)
        new = new.pivot_table(index=["region_name", "date"], columns="var", values="value").reset_index()
        new["date"] = pd.to_datetime(new["date"])
        new = new.dropna(subset=["prcp", "tmax", "tmin"])
        hist = hist[hist["date"] < pd.Timestamp(str(need[0].start_time.date()))]
        hist = pd.concat([hist, new[["region_name", "date", "tmax", "tmin", "prcp"]]], ignore_index=True)
    hist = hist.sort_values(["region_name", "date"]).drop_duplicates(["region_name", "date"], keep="last")
    hist.to_csv(out_csv, index=False)
    print(f"Weather: {hist['region_name'].nunique()} counties, {hist['date'].min().date()} to {hist['date'].max().date()}")
    return hist

# ----------------------------------------------------------------------------- forecast


def forecast(usdm_csv, weather_csv, out_dir="forecasts", leads=(1, 2, 4), seed=0) -> pd.DataFrame:
    """Train on all history, forecast from the latest Drought Monitor week that has weather."""
    levels = usdm_levels(pd.read_csv(usdm_csv, dtype={"FIPS": str}))
    met = region_weather(pd.read_csv(weather_csv))
    X = build_features(met, None, ref_years=(1991, 2020))
    weather_cols = [c for c in X.columns if c not in ("site", "date")]
    cols = weather_cols + FEATURES_NOW

    # Forecast from the newest map. Weather runs ~4 days behind, so if it stops before the
    # map date, every row (training too) uses weather from that many days before its map.
    issue = levels["date"].max()
    gap = max(0, (issue - X.dropna(subset=["prcp_365d"])["date"].max()).days)
    if gap:
        X = X.assign(date=X["date"] + pd.Timedelta(days=gap))
        print(f"Weather ends {gap} days before the {issue.date()} map; using weather from {gap} days "
              f"before each map, in training and in the forecast.")
    now = current_state(levels).merge(X, on=["site", "date"], how="inner")
    now = now[now["date"] == issue]
    made_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = []
    for lead in leads:
        T = make_table(levels, X, lead)
        m = _rf(seed).fit(T[cols].fillna(T[cols].median()), T["y"])
        P = m.predict_proba(now[cols].fillna(T[cols].median()))
        full = np.zeros((len(now), 4))
        for j, c in enumerate(m.classes_):
            full[:, c] = P[:, j]
        for i, (_, r) in enumerate(now.iterrows()):
            p_d1 = full[i, 2:].sum()
            rows.append({
                "made_at": made_at, "issue_date": issue.date(), "lead_weeks": lead,
                "target_date": (issue + pd.Timedelta(weeks=lead)).date(), "region": r["site"],
                "level_now": LEVELS[int(r["level_now"])], "forecast_level": LEVELS[int(full[i].argmax())],
                **{f"p_{lv}": round(full[i, k], 3) for k, lv in enumerate(LEVELS)},
                "p_d1plus": round(p_d1, 3),
                "new_drought_warning": bool(r["level_now"] < 2 and p_d1 >= ALARM),
                "weather_through": (issue - pd.Timedelta(days=gap)).date(),
                "trained_through": T["target_date"].max().date(), "train_rows": len(T),
            })
    F = pd.DataFrame(rows)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    F.to_csv(out / "latest.csv", index=False)
    log_f = out / "log.csv"
    age = (pd.Timestamp.now().normalize() - issue).days
    if age > 9:
        # an old map means the newest one was not downloaded; its targets may already be
        # known, so it would not be an honest live test
        print(f"\nThe newest map available here is {age} days old, so this forecast is NOT added to "
              f"{log_f} (only to latest.csv). Check that the Drought Monitor download worked.")
        F2 = pd.read_csv(log_f, keep_default_na=False, na_values=[""]) if log_f.exists() else None
    elif log_f.exists():
        old = pd.read_csv(log_f, keep_default_na=False, na_values=[""])  # "None" is a level, not missing
        F2 = pd.concat([old, F.astype({"issue_date": str, "target_date": str, "weather_through": str, "trained_through": str})],
                       ignore_index=True)
        F2 = F2.drop_duplicates(["issue_date", "lead_weeks", "region"], keep="first")  # never rewrite history
    else:
        F2 = F
    if F2 is not None:
        F2.to_csv(log_f, index=False)
    print(f"\nForecast from the {issue.date()} Drought Monitor map (weather through {(issue - pd.Timedelta(days=gap)).date()}):")
    show = F.pivot_table(index="region", columns="lead_weeks", values="forecast_level", aggfunc="first")
    show.columns = [f"{c} wk" for c in show.columns]
    show.insert(0, "now", F.groupby("region")["level_now"].first())
    pd_show = F.pivot_table(index="region", columns="lead_weeks", values="p_d1plus")
    pd_show.columns = [f"P(D1+) {c} wk" for c in pd_show.columns]
    print(pd.concat([show, pd_show], axis=1).to_string())
    w = F[F["new_drought_warning"]]
    print("\nNew-drought warnings: " + (", ".join(f"{r.region} ({r.lead_weeks} wk, {r.p_d1plus:.0%})"
                                                for r in w.itertuples()) if len(w) else "none"))
    return F

# ----------------------------------------------------------------------------- verify


def verify(usdm_csv, out_dir="forecasts") -> pd.DataFrame | None:
    """Score every logged forecast whose target week has been mapped since."""
    log_f = Path(out_dir) / "log.csv"
    if not log_f.exists():
        print("No forecast log yet."); return None
    L = pd.read_csv(log_f, parse_dates=["issue_date", "target_date"], keep_default_na=False, na_values=[""])
    lv = usdm_levels(pd.read_csv(usdm_csv, dtype={"FIPS": str}))[["region", "date", "level"]]
    V = L.merge(lv.rename(columns={"date": "target_date", "level": "actual"}), on=["region", "target_date"], how="inner")
    if V.empty:
        print(f"{len(L)} forecasts logged; none has reached its target week yet.")
        return V
    V["actual_level"] = V["actual"].map(dict(enumerate(LEVELS)))
    code = {lv: i for i, lv in enumerate(LEVELS)}
    rows = []
    for lead, g in V.groupby("lead_weeks"):
        now, y = g["level_now"].map(code), g["actual"]
        quiet = now < 2
        rows.append({"lead_weeks": lead, "forecasts_checked": len(g), "issue_weeks": g["issue_date"].nunique(),
                     **level_metrics(y, g["forecast_level"].map(code)),
                     "persistence_exact": float((now == y).mean()),
                     **{k: v for k, v in binary_metrics(y >= 2, g["p_d1plus"], ALARM).items()
                        if k in ("hit_rate", "false_alarm_ratio", "brier")},
                     "new_droughts": int((quiet & (y >= 2)).sum()),
                     "new_droughts_warned": int((quiet & (y >= 2) & g["new_drought_warning"]).sum())})
    S = pd.DataFrame(rows)
    V.to_csv(Path(out_dir) / "verification.csv", index=False)
    S.to_csv(Path(out_dir) / "verification_summary.csv", index=False)
    print("\nLive record so far (forecasts made before the answer was known):")
    print(S.round(3).to_string(index=False))
    return S


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m drought_rf.live", description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["all", "update", "forecast", "verify"], nargs="?", default="all")
    ap.add_argument("--usdm", default="data/raw/usdm_ct_live.csv")
    ap.add_argument("--weather", default="data/raw/ct_county_daily_live.csv")
    ap.add_argument("--fallback-usdm", default="../data/ct_drought_monitor_2000_2026.csv",
                    help="used if the Drought Monitor service can't be reached")
    ap.add_argument("--out", default="forecasts")
    ap.add_argument("--leads", default="1,2,4")
    a = ap.parse_args(argv)
    if a.step in ("all", "update"):
        fetch_usdm(a.usdm, a.fallback_usdm)
        fetch_weather(a.weather)
    if a.step in ("all", "forecast"):
        forecast(a.usdm, a.weather, a.out, tuple(int(x) for x in a.leads.split(",")))
    if a.step in ("all", "verify"):
        verify(a.usdm, a.out)


if __name__ == "__main__":
    main()
