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


def test_usdm_levels_and_lead_alignment():
    from drought_rf.usdm import hargreaves_pet, make_table, usdm_levels
    dates = pd.date_range("2020-01-07", periods=12, freq="7D")
    d1 = [0, 0, 0, 10, 60, 80, 90, 100, 100, 40, 0, 0]
    u = pd.DataFrame({"County": "Capitol Planning Region", "ValidStart": dates.strftime("%Y-%m-%d"),
                      "D0": [max(v, 55 if i in (2, 3) else 0) for i, v in enumerate(d1)], "D1": d1,
                      "D2": [0] * 6 + [70, 70, 0, 0, 0, 0], "D3": 0, "D4": 0})
    lv = usdm_levels(u)
    assert lv["level"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 2, 0, 0, 0]
    X = pd.DataFrame({"site": "Capitol", "date": dates, "f": range(12)})
    T = make_table(lv, X, lead_weeks=2)
    assert (T["y"].values == lv["level"].values[2:]).all()
    assert (T["level_now"].values == lv["level"].values[:-2]).all()
    assert (hargreaves_pet(np.array([30.0]), np.array([18.0]), np.array([190])) > 3).all()


def test_scorecard_counts_and_skill():
    from drought_rf.scorecard import binary_metrics, bootstrap_ci, level_metrics, reliability
    y = np.array([1, 1, 0, 0, 0, 0, 1, 0])
    p = np.array([.9, .2, .6, .1, .1, .1, .8, .1])
    s = binary_metrics(y, p, 0.5, p_ref=np.full(8, y.mean()))
    assert (s["hits"], s["misses"], s["false_alarms"], s["correct_quiet"]) == (2, 1, 1, 4)
    assert abs(s["hit_rate"] - 2 / 3) < 1e-9 and abs(s["false_alarm_ratio"] - 1 / 3) < 1e-9
    assert s["brier_skill"] > 0  # sharper than the base rate here
    assert level_metrics([0, 1, 2, 3], [0, 2, 2, 1])["mean_level_error"] == 0.75
    r = reliability(y, p)
    assert r["n"].sum() == 8
    df = pd.DataFrame({"y": np.tile(y, 5), "p": np.tile(p, 5), "year": np.repeat(range(5), 8)})
    ci = bootstrap_ci(df, lambda d: binary_metrics(d["y"], d["p"]), n_boot=20)
    assert ci.loc["hit_rate", "low"] <= ci.loc["hit_rate", "high"]


def test_live_log_keeps_first_forecast(tmp_path):
    # verify() must read "None" as a level, and the log must score against later maps
    from drought_rf.live import verify
    log = pd.DataFrame({"made_at": "x", "issue_date": "2020-01-07", "lead_weeks": 1, "target_date": "2020-01-14",
                        "region": ["Capitol"], "level_now": ["None"], "forecast_level": ["None"],
                        "p_d1plus": [0.05], "new_drought_warning": [False]})
    log.to_csv(tmp_path / "log.csv", index=False)
    usdm = pd.DataFrame({"County": "Capitol Planning Region", "ValidStart": ["2020-01-07", "2020-01-14"],
                         "D0": 0.0, "D1": 0.0, "D2": 0.0})
    usdm.to_csv(tmp_path / "usdm.csv", index=False)
    S = verify(tmp_path / "usdm.csv", tmp_path)
    assert S.loc[0, "exact_level"] == 1.0 and S.loc[0, "forecasts_checked"] == 1
