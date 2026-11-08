import numpy as np
import pandas as pd

from drought_rf import synthetic
from drought_rf.donor import DonorSelector
from drought_rf.features import build_features, make_supervised
from drought_rf.percentiles import variable_threshold_percentile
from drought_rf.pipeline import run
from drought_rf.splits import walk_forward
from drought_rf.targets import drought_label, streamflow_percentiles


def test_weibull_uniform():
    idx = pd.date_range("1990-01-01", "2019-12-31")
    s = pd.Series(np.random.default_rng(0).normal(size=len(idx)), index=idx)
    p = variable_threshold_percentile(s)
    assert p.between(0, 1, inclusive="neither").all()
    assert abs((p < 0.2).mean() - 0.2) < 0.01


def test_seasonal_signal_removed():
    # a pure seasonal cycle should not look like drought every summer
    idx = pd.date_range("1990-01-01", "2019-12-31")
    s = pd.Series(np.sin(2 * np.pi * idx.dayofyear / 365) + np.random.default_rng(1).normal(0, .1, len(idx)), index=idx)
    p = variable_threshold_percentile(s)
    assert abs(p[idx.month == 7].mean() - 0.5) < 0.05


def test_lead_alignment():
    met, flow, tele, _ = synthetic.make(n_sites=1, start="2000-01-01", end="2003-12-31")
    fp = streamflow_percentiles(flow); fp["drought"] = drought_label(fp["q_pct"])
    X = build_features(met, tele, percentiles=False)
    D = make_supervised(X, fp, 30)
    row = D.iloc[500]
    truth = fp.set_index("date").loc[row["date"] + pd.Timedelta(days=30), "drought"]
    assert row["y"] == truth and (D["target_date"] - D["date"]).dt.days.eq(30).all()


def test_walk_forward_embargo():
    d = pd.DataFrame({"date": pd.date_range("1990-01-01", "2010-12-31")})
    d["target_date"] = d["date"] + pd.Timedelta(days=30)
    for _, tr, te in walk_forward(d, 2005, 2006, embargo_days=30):
        assert d.loc[tr, "target_date"].max() < d.loc[te, "date"].min() - pd.Timedelta(days=29)


def test_end_to_end_small(tmp_path):
    P, S = run(None, tmp_path, leads=(7,), models=("rf", "climatology"), verbose=False)
    rf = S[S.model == "rf"]
    assert (rf["kappa"] > 0.2).all() and (rf["brier_skill"] > 0).all()


def test_donor_selector():
    rng = np.random.default_rng(0)
    sites = [f"s{i}" for i in range(20)]
    static = pd.DataFrame(rng.normal(size=(20, 3)), index=sites, columns=list("abc"))
    imp = pd.DataFrame(static.values @ rng.normal(size=(3, 8)) + rng.normal(0, .1, (20, 8)), index=sites)
    sel = DonorSelector().fit(imp, static, pd.Series(0.6, index=sites))
    d = sel.donors(static.loc["s3"], n=3)
    assert d.index[0] == "s3"
