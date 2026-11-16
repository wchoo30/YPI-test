"""How good is the model? Error metrics, with uncertainty, for both drought models.

Reads the predictions.csv files the models write and scores every test week or day:

  * yes/no scores for "drought" (D1 or worse for the Drought Monitor model, flow below the
    20th percentile for the gauge model): hit rate, false alarm ratio, false alarm rate,
    critical success index, Peirce skill, Cohen's kappa;
  * probability scores: Brier score, Brier skill against climatology, ROC AUC, and a
    reliability table (when the model says 30 %, does drought happen about 30 % of the time?);
  * for the Drought Monitor model, level scores too: exact level, within one level, mean
    level error, weighted kappa;
  * 90 % confidence intervals from a block bootstrap that resamples whole test years, since
    drought weeks inside one year are not independent;
  * scores per year and per region/gauge, and a threshold table showing the trade-off
    between catching droughts and raising false alarms.

    python -m drought_rf.scorecard --usdm outputs/usdm/predictions.csv
    python -m drought_rf.scorecard --gauge outputs/predictions.csv

Results go to outputs/scorecard/ (usdm_*.csv, gauge_*.csv).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

# ----------------------------------------------------------------------------- yes/no + probability


def binary_metrics(y, p, threshold: float = 0.5, p_ref=None) -> dict:
    """y: 1 = drought happened. p: forecast probability of drought. p_ref: reference
    (climatology) probability for the Brier skill score."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    yhat = (p >= threshold).astype(int)
    hits = int(((y == 1) & (yhat == 1)).sum()); misses = int(((y == 1) & (yhat == 0)).sum())
    fa = int(((y == 0) & (yhat == 1)).sum()); cn = int(((y == 0) & (yhat == 0)).sum())
    div = lambda a, b: a / b if b else np.nan
    pod, pofd = div(hits, hits + misses), div(fa, fa + cn)
    out = {
        "n": len(y), "drought_share": float(y.mean()) if len(y) else np.nan,
        "hits": hits, "misses": misses, "false_alarms": fa, "correct_quiet": cn,
        "accuracy": div(hits + cn, len(y)),
        "hit_rate": pod,                                   # share of droughts that were called
        "false_alarm_ratio": div(fa, hits + fa),           # share of drought calls that were wrong
        "false_alarm_rate": pofd,                          # share of quiet times wrongly called
        "precision": div(hits, hits + fa),                 # = 1 - false alarm ratio
        "recall": pod,                                     # = hit rate
        "f1": div(2 * hits, 2 * hits + misses + fa),
        "csi": div(hits, hits + misses + fa),              # critical success index
        "peirce_skill": pod - pofd if not (np.isnan(pod) or np.isnan(pofd)) else np.nan,
        "kappa": float(cohen_kappa_score(y, yhat)) if len(set(y)) > 1 or len(set(yhat)) > 1 else np.nan,
        "brier": float(np.mean((p - y) ** 2)) if len(y) else np.nan,
        "prob_rmse": float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else np.nan,  # RMSE of the probability = sqrt(Brier)
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 and len(set(p)) > 1 else np.nan,
    }
    if p_ref is not None:
        bref = float(np.mean((np.clip(np.asarray(p_ref, dtype=float), 0, 1) - y) ** 2))
        out["brier_climatology"] = bref
        out["brier_skill"] = 1 - out["brier"] / bref if bref > 0 else np.nan
    return out


def level_metrics(y, pred) -> dict:
    """Scores for a forecast level 0..3 (None, D0, D1, D2+)."""
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    return {
        "exact_level": float((y == pred).mean()),
        "within_one_level": float((np.abs(y - pred) <= 1).mean()),
        "mean_level_error": float(np.abs(y - pred).mean()),       # MAE, in levels
        "rmse_levels": float(np.sqrt(((y - pred) ** 2).mean())),  # RMSE, in levels (big misses count more)
        # MAPE divides by the actual level, so weeks at None (level 0, most weeks) are left out;
        # read it as "% error on weeks that were at least abnormally dry"
        "mape_dry_weeks": float((np.abs(y - pred)[y > 0] / y[y > 0]).mean() * 100) if (y > 0).any() else np.nan,
        "bias_levels": float((pred - y).mean()),                   # > 0: forecasts too dry
        "kappa_weighted": float(cohen_kappa_score(y, pred, weights="quadratic"))
        if len(set(y)) > 1 else np.nan,
    }


def reliability(y, p, bins=(0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001)) -> pd.DataFrame:
    """Forecast probability bins vs how often drought actually happened in each bin."""
    d = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p, dtype=float)})
    d["bin"] = pd.cut(d["p"], bins, right=False)
    r = d.groupby("bin", observed=True).agg(n=("y", "size"), mean_forecast=("p", "mean"), observed=("y", "mean"))
    return r.reset_index().assign(bin=lambda x: x["bin"].astype(str))


def threshold_table(y, p, grid=np.round(np.arange(0.1, 0.91, 0.1), 2)) -> pd.DataFrame:
    """Hit rate and false alarms at each alarm threshold. Shows the trade-off; choose the
    operating threshold from earlier years, not from this table's test years."""
    return pd.DataFrame([{"threshold": t, **{k: v for k, v in binary_metrics(y, p, t).items()
                                              if k in ("hit_rate", "false_alarm_ratio", "false_alarm_rate", "csi", "kappa")}}
                         for t in grid])


def bootstrap_ci(df: pd.DataFrame, score_fn, by: str = "year", n_boot: int = 300,
                 level: float = 0.90, seed: int = 0) -> pd.DataFrame:
    """Block bootstrap: resample whole years with replacement, recompute every score.
    Returns rows = score name, columns = low, high."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(by)}
    keys = list(groups)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        draws.append(score_fn(pd.concat([groups[keys[i]] for i in pick], ignore_index=True)))
    D = pd.DataFrame(draws).apply(pd.to_numeric, errors="coerce")
    a = (1 - level) / 2
    return pd.DataFrame({"low": D.quantile(a), "high": D.quantile(1 - a)})


def _with_ci(df, score_fn, by="year", n_boot=300, keep=None) -> dict:
    est = score_fn(df)
    ci = bootstrap_ci(df, score_fn, by, n_boot)
    for k in keep or est:
        if k in ci.index:
            est[f"{k}_lo"], est[f"{k}_hi"] = ci.loc[k, "low"], ci.loc[k, "high"]
    return est

# ----------------------------------------------------------------------------- Drought Monitor model


USDM_MODELS = ["climatology", "persistence", "rf_weather", "rf_weather_plus_now"]
USDM_CI = ["exact_level", "mean_level_error", "rmse_levels", "f1", "kappa_weighted", "hit_rate", "false_alarm_ratio",
           "brier_skill", "roc_auc", "onset_caught", "onset_false_alarm_rate"]


def _usdm_prob(g: pd.DataFrame, model: str) -> np.ndarray:
    col = model + "_p_d1plus"
    if col in g:
        return g[col].to_numpy()
    if model == "climatology" and "clim_prob_d1plus" in g:
        return g["clim_prob_d1plus"].to_numpy()
    return (g[model] >= 2).astype(float).to_numpy()


def _usdm_ref(g: pd.DataFrame) -> np.ndarray:
    if "clim_prob_d1plus" in g:
        return g["clim_prob_d1plus"].to_numpy()
    # older outputs: fall back to the overall D1+ share (a slightly easier reference)
    return np.full(len(g), float((g["y"] >= 2).mean()))


def _usdm_scores(g: pd.DataFrame, model: str, alarm: float) -> dict:
    y = g["y"].to_numpy()
    p = _usdm_prob(g, model)
    thr = alarm if (model + "_p_d1plus") in g else 0.5
    s = {**level_metrics(y, g[model]), **binary_metrics(y >= 2, p, thr, _usdm_ref(g))}
    quiet_now = g["level_now"].to_numpy() < 2
    on, calm = quiet_now & (y >= 2), quiet_now & (y < 2)
    s["onsets"] = int(on.sum())
    s["onset_caught"] = float((p[on] >= thr).mean()) if on.any() else np.nan
    s["onset_false_alarm_rate"] = float((p[calm] >= thr).mean()) if calm.any() else np.nan
    return s


def usdm_scorecard(P: pd.DataFrame, alarm: float = 0.3, n_boot: int = 300) -> dict[str, pd.DataFrame]:
    """P: outputs/usdm/predictions.csv. D1+ alarm for the forests = P(D1 or worse) >= alarm."""
    P = P.copy()
    P["year"] = pd.to_datetime(P["target_date"]).dt.year
    models = [m for m in USDM_MODELS if m in P]
    card, rel, thr, yr, reg = [], [], [], [], []
    for lead, g in P.groupby("lead_weeks"):
        for m in models:
            fn = lambda d, m=m: _usdm_scores(d, m, alarm)
            card.append({"lead_weeks": lead, "model": m, **_with_ci(g, fn, "year", n_boot, USDM_CI)})
            for y_, d in g.groupby("year"):
                yr.append({"lead_weeks": lead, "model": m, "year": y_, **fn(d)})
            for r_, d in g.groupby("site"):
                reg.append({"lead_weeks": lead, "model": m, "region": r_, **fn(d)})
            if (m + "_p_d1plus") in g:
                rel.append(reliability(g["y"] >= 2, g[m + "_p_d1plus"]).assign(lead_weeks=lead, model=m))
                thr.append(threshold_table(g["y"] >= 2, g[m + "_p_d1plus"]).assign(lead_weeks=lead, model=m))
    cat = lambda L: pd.concat(L, ignore_index=True) if L else pd.DataFrame()
    return {"scorecard": pd.DataFrame(card), "reliability": cat(rel), "thresholds": cat(thr),
            "by_year": pd.DataFrame(yr), "by_region": pd.DataFrame(reg)}

# ----------------------------------------------------------------------------- gauge model


GAUGE_CI = ["f1", "kappa", "hit_rate", "false_alarm_ratio", "brier_skill", "roc_auc"]
ID = {"site", "date", "target_date", "y", "lead", "fold", "year"}


def gauge_scorecard(P: pd.DataFrame, threshold: float = 0.5, n_boot: int = 300) -> dict[str, pd.DataFrame]:
    """P: outputs/predictions.csv from python -m drought_rf (one probability column per model)."""
    P = P.copy()
    P["year"] = pd.to_datetime(P["target_date"]).dt.year
    models = [c for c in P.columns if c not in ID and pd.api.types.is_numeric_dtype(P[c])]
    ref = "climatology" if "climatology" in P else None
    card, rel, thr, yr, site = [], [], [], [], []
    for lead, g in P.groupby("lead"):
        for m in models:
            fn = lambda d, m=m: binary_metrics(d["y"], d[m], threshold, d[ref] if ref else None)
            card.append({"lead": lead, "model": m, **_with_ci(g, fn, "year", n_boot, GAUGE_CI)})
            for y_, d in g.groupby("year"):
                yr.append({"lead": lead, "model": m, "year": y_, **fn(d)})
            for s_, d in g.groupby("site"):
                site.append({"lead": lead, "model": m, "site": s_, **fn(d)})
            rel.append(reliability(g["y"], g[m]).assign(lead=lead, model=m))
            thr.append(threshold_table(g["y"], g[m]).assign(lead=lead, model=m))
    return {"scorecard": pd.DataFrame(card), "reliability": pd.concat(rel, ignore_index=True),
            "thresholds": pd.concat(thr, ignore_index=True),
            "by_year": pd.DataFrame(yr), "by_site": pd.DataFrame(site)}

# ----------------------------------------------------------------------------- plain-language summary


def _pct(x):
    return "n/a" if pd.isna(x) else f"{100 * x:.0f}%"


def _ci(row, k):
    lo, hi = row.get(f"{k}_lo"), row.get(f"{k}_hi")
    return "" if lo is None or pd.isna(lo) else f" ({100 * lo:.0f}-{100 * hi:.0f}%)"


def describe_usdm(card: pd.DataFrame) -> str:
    lines = ["Drought Monitor model, best forest vs baselines (90% intervals in brackets):"]
    for lead, g in card.groupby("lead_weeks"):
        g = g.set_index("model")
        f = g.loc["rf_weather_plus_now"] if "rf_weather_plus_now" in g.index else None
        ps = g.loc["persistence"] if "persistence" in g.index else None
        if f is None:
            continue
        lines.append(
            f"  {lead} wk ahead: exact level {_pct(f.exact_level)}{_ci(f, 'exact_level')}"
            f" vs persistence {_pct(ps.exact_level) if ps is not None else 'n/a'};"
            f" D1+ caught {_pct(f.hit_rate)}, wrong D1+ calls {_pct(f.false_alarm_ratio)};"
            f" Brier skill {f.brier_skill:+.2f}; new droughts caught {_pct(f.onset_caught)}{_ci(f, 'onset_caught')}"
            f" with {_pct(f.onset_false_alarm_rate)} false alarms;"
            f" RMSE {f.rmse_levels:.2f} levels (persistence {ps.rmse_levels:.2f}), MAE {f.mean_level_error:.2f},"
            f" F1 {f.f1:.2f} (persistence {ps.f1:.2f})")
    return "\n".join(lines)


def describe_gauge(card: pd.DataFrame) -> str:
    lines = ["Gauge model (90% intervals in brackets):"]
    for (lead, m), r in card.set_index(["lead", "model"]).iterrows():
        bs = "" if pd.isna(r.get("brier_skill")) else f", Brier skill {r.brier_skill:+.2f}"
        lines.append(f"  {lead:>2} d, {m:<12} F1 {r.f1:.2f}, precision {r.precision:.2f}, recall {r.recall:.2f},"
                     f" probability RMSE {r.prob_rmse:.2f}, kappa {r.kappa:.2f}, droughts caught {_pct(r.hit_rate)}{_ci(r, 'hit_rate')},"
                     f" wrong calls {_pct(r.false_alarm_ratio)}, AUC {r.roc_auc:.2f}{bs}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m drought_rf.scorecard", description=__doc__.split("\n\n")[0])
    ap.add_argument("--usdm", help="outputs/usdm/predictions.csv")
    ap.add_argument("--gauge", help="outputs/predictions.csv")
    ap.add_argument("--out", default="outputs/scorecard")
    ap.add_argument("--alarm", type=float, default=0.3, help="D1+ alarm threshold for the Drought Monitor forests")
    ap.add_argument("--threshold", type=float, default=0.5, help="drought threshold for the gauge model")
    ap.add_argument("--boot", type=int, default=300, help="bootstrap resamples (fewer = faster)")
    a = ap.parse_args(argv)
    if not (a.usdm or a.gauge):
        ap.error("give --usdm and/or --gauge")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.usdm:
        R = usdm_scorecard(pd.read_csv(a.usdm), a.alarm, a.boot)
        for k, v in R.items():
            v.to_csv(out / f"usdm_{k}.csv", index=False)
        print(describe_usdm(R["scorecard"]))
    if a.gauge:
        R = gauge_scorecard(pd.read_csv(a.gauge), a.threshold, a.boot)
        for k, v in R.items():
            v.to_csv(out / f"gauge_{k}.csv", index=False)
        print(describe_gauge(R["scorecard"]))
    print(f"Tables written to {out}/")


if __name__ == "__main__":
    main()
