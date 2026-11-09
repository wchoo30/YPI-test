"""End-to-end run: labels -> predictors -> one model per site and lead -> scores."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import synthetic
from .evaluate import scores
from .features import add_antecedent_flow, build_features, make_supervised
from .io import load
from .models import MODELS
from .splits import paper_split, walk_forward
from .targets import drought_label, streamflow_percentiles

ID_COLS = ["site", "date", "target_date", "y"]
FLOW_COLS = ["qpct_0d", "qpct_7d", "qpct_30d"]


def run(data_dir=None, out_dir="outputs", leads=(0, 7, 14, 30),
        models=("rf", "lgbm", "logistic", "climatology", "persistence"),
        split="paper", use_flow=False, threshold=0.5, first_test_year=2011, last_test_year=2020,
        verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    if data_dir is None:
        log("No data folder given: using SYNTHETIC data (pipeline check only, not results).")
        met, flow, tele, static = synthetic.make()
    else:
        met, flow, tele, static = load(data_dir)

    ref_years = (1985, 2015) if split == "paper" else (int(met["date"].dt.year.min()), first_test_year - 1)
    log(f"Percentile reference years: {ref_years}")
    fp = streamflow_percentiles(flow, ref_years=ref_years)
    fp["drought"] = drought_label(fp["q_pct"])
    X = build_features(met, tele, ref_years=ref_years)
    X = add_antecedent_flow(X, fp)
    feat_cols = [c for c in X.columns if c not in ID_COLS and (use_flow or c not in FLOW_COLS)]
    log(f"{X['site'].nunique()} sites, {len(feat_cols)} predictors")

    preds, rows, imps = [], [], []
    for lead in leads:
        D = make_supervised(X, fp, lead)
        for site, g in D.groupby("site"):
            g = g.reset_index(drop=True)
            folds = paper_split(g) if split == "paper" else walk_forward(g, first_test_year, last_test_year)
            for fold, tr, te in folds:
                if tr.sum() < 365 or te.sum() == 0:
                    continue
                Xtr, Xte = g.loc[tr], g.loc[te]
                out = Xte[["site", "date", "target_date", "y"]].copy()
                out["lead"], out["fold"] = lead, fold
                for name in models:
                    m = MODELS[name]()
                    if name in ("climatology", "persistence"):
                        m.fit(Xtr, Xtr["y"]); out[name] = m.predict_proba(Xte)
                    else:
                        m.fit(Xtr[feat_cols], Xtr["y"].astype(int)); out[name] = m.predict_proba(Xte[feat_cols])
                        if name == "rf":
                            imps.append(m.feature_importances_.rename(f"{site}|{lead}|{fold}"))
                preds.append(out)
        log(f"lead {lead:>2} d done")

    P = pd.concat(preds, ignore_index=True)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / "predictions.csv", index=False)  # saved before scoring, so a scoring error loses nothing
    for (site, lead), g in P.groupby(["site", "lead"]):
        for name in models:
            if g[name].isna().all():
                continue
            s = scores(g["y"], g[name], threshold=threshold, p_clim=g.get("climatology"))
            rows.append({"site": site, "lead": lead, "model": name, **s})
    S = pd.DataFrame(rows)
    S.to_csv(out / "scores.csv", index=False)
    if imps:
        I = pd.concat(imps, axis=1)
        I.to_csv(out / "rf_importance.csv")
    summary = S.groupby(["lead", "model"])[["kappa", "sensitivity", "specificity", "brier", "brier_skill"]].mean()
    summary.to_csv(out / "summary.csv")
    log(summary.round(3).to_string())
    return P, S
