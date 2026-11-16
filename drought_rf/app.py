"""Connecticut drought forecast viewer.

    streamlit run app.py          (from the drought_rf folder)

Shows this week's Drought Monitor forecast for each planning region (1, 2 and 4 weeks ahead),
past forecasts next to what really happened, and the live track record. Reads:
  forecasts/latest.csv, forecasts/log.csv          from python -m drought_rf.live
  outputs/usdm/predictions.csv                      from python -m drought_rf.usdm (the 2010-2025 test)
  data/raw/usdm_ct_live.csv (or ../data/...csv)     actual Drought Monitor levels
The "Update data and forecast" button in the sidebar runs python -m drought_rf.live for you.
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from drought_rf.scorecard import binary_metrics, level_metrics
from drought_rf.usdm import LEVELS, usdm_levels

HERE = Path(__file__).resolve().parent
FC = HERE / "forecasts"
BACKTEST = HERE / "outputs" / "usdm" / "predictions.csv"
USDM_LIVE = HERE / "data" / "raw" / "usdm_ct_live.csv"
USDM_REPO = HERE.parent / "data" / "ct_drought_monitor_2000_2026.csv"
WEATHER = HERE / "data" / "raw" / "ct_county_daily_live.csv"
ALARM = 0.3
CODE = {lv: i for i, lv in enumerate(LEVELS)}
# Drought Monitor's own map colours (D2+ uses the D2 colour)
COLOR = {"None": "#e9efe6", "D0": "#ffff00", "D1": "#fcd37f", "D2+": "#ffaa00"}
TEXT = {"None": "#33402f", "D0": "#3d3a00", "D1": "#4a3200", "D2+": "#3d2200"}
WORDS = {"None": "No drought", "D0": "Abnormally dry", "D1": "Moderate drought", "D2+": "Severe or worse"}

st.set_page_config(page_title="CT drought forecast", layout="wide")


@st.cache_data(show_spinner=False)
def load_csv(path: str, mtime: float, **kw) -> pd.DataFrame:  # mtime busts the cache when the file changes
    return pd.read_csv(path, **kw)


def read(path: Path, **kw):
    return load_csv(str(path), path.stat().st_mtime, **kw) if path.exists() else None


def actual_levels() -> pd.DataFrame | None:
    src = USDM_LIVE if USDM_LIVE.exists() else USDM_REPO
    d = read(src, dtype={"FIPS": str})
    if d is None:
        return None
    lv = usdm_levels(d)
    return lv.rename(columns={"region": "region", "date": "target_date", "level": "actual"})[["region", "target_date", "actual"]]


def past_forecasts() -> pd.DataFrame:
    """Backtest forecasts (2010-2025 test) and live logged forecasts in one table."""
    parts = []
    bt = read(BACKTEST, parse_dates=["date", "target_date"])
    if bt is not None:
        b = bt[bt["lead_weeks"] > 0]
        parts.append(pd.DataFrame({
            "source": "Backtest", "region": b["site"], "issue_date": b["date"], "target_date": b["target_date"],
            "lead_weeks": b["lead_weeks"], "level_now": b["level_now"], "forecast": b["rf_weather_plus_now"],
            "p_d1plus": b["rf_weather_plus_now_p_d1plus"], "actual": b["y"]}))
    log = read(FC / "log.csv", parse_dates=["issue_date", "target_date"], keep_default_na=False, na_values=[""])
    act = actual_levels()
    if log is not None and act is not None and len(log):
        L = log.merge(act, on=["region", "target_date"], how="left")
        parts.append(pd.DataFrame({
            "source": "Live", "region": L["region"], "issue_date": L["issue_date"], "target_date": L["target_date"],
            "lead_weeks": L["lead_weeks"], "level_now": L["level_now"].map(CODE), "forecast": L["forecast_level"].map(CODE),
            "p_d1plus": L["p_d1plus"], "actual": L["actual"]}))
    if not parts:
        return pd.DataFrame()
    P = pd.concat(parts, ignore_index=True)
    return P


def level_cell(level: str, p: float | None = None) -> str:
    sub = "" if p is None else f"<div style='font-size:12px;opacity:.85'>D1+ chance {p:.0%}</div>"
    return (f"<div style='background:{COLOR[level]};color:{TEXT[level]};border-radius:8px;padding:8px 10px;"
            f"min-width:90px'><b>{level}</b>{sub}</div>")


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    latest = read(FC / "latest.csv", keep_default_na=False, na_values=[""])
    if latest is not None and len(latest):
        st.write(f"Forecast from the **{latest['issue_date'].iloc[0]}** Drought Monitor map")
        if "weather_through" in latest:
            st.write(f"Weather through **{latest['weather_through'].iloc[0]}**")
        st.caption(f"Made {latest['made_at'].iloc[0]}")
    else:
        st.write("No forecast yet.")
    if st.button("Update data and forecast", type="primary", use_container_width=True):
        from drought_rf import live
        with st.spinner("Downloading the newest map and weather, then forecasting (about a minute)..."):
            try:
                live.fetch_usdm(USDM_LIVE, USDM_REPO)
                live.fetch_weather(WEATHER, cache_dir=HERE / "data" / "raw" / "_nclimgrid_cache")
                live.forecast(USDM_LIVE, WEATHER, FC)
                live.verify(USDM_LIVE, FC)
                st.success("Updated.")
            except Exception as e:  # show the reason instead of a stack trace
                st.error(f"Update failed: {e}")
        st.rerun()
    st.caption("Same as running `python -m drought_rf.live` in the drought_rf folder.")
    st.divider()
    st.caption("Levels: None = no drought, D0 = abnormally dry, D1 = moderate drought, D2+ = severe or worse. "
               "A region's level is the worst one covering at least half of it.")

st.title("Connecticut drought forecast")
tab_now, tab_past, tab_live = st.tabs(["This week's forecast", "Past forecasts vs what happened", "Live track record"])

# ----------------------------------------------------------------------------- this week
with tab_now:
    if latest is None or not len(latest):
        st.info("No forecast yet. Click **Update data and forecast** in the sidebar, or run `python -m drought_rf.live`.")
    else:
        issue = pd.Timestamp(latest["issue_date"].iloc[0])
        age = (pd.Timestamp.now().normalize() - issue).days
        if age > 9:
            st.warning(f"This forecast starts from a map that is {age} days old, so some of its target weeks have "
                       "already passed. Click **Update data and forecast** to get the newest map.")
        leads = sorted(latest["lead_weeks"].unique())
        targets = {l: latest.loc[latest["lead_weeks"] == l, "target_date"].iloc[0] for l in leads}
        warn = latest[latest["new_drought_warning"].astype(str).str.lower() == "true"]
        n_d1 = latest[latest["forecast_level"].isin(["D1", "D2+"])]
        c1, c2, c3 = st.columns(3)
        c1.metric("Regions in D1+ now", int(latest.drop_duplicates("region")["level_now"].isin(["D1", "D2+"]).sum()), help="Out of 9 planning regions")
        c2.metric(f"Forecast in D1+ in {leads[0]} wk", int(n_d1[n_d1["lead_weeks"] == leads[0]]["region"].nunique()))
        c3.metric("New-drought warnings", len(warn), help=f"Region below D1 now with at least a {ALARM:.0%} chance of D1+ by the target week")

        head = "".join(f"<th style='text-align:left;padding:6px'>{h}</th>" for h in
                       ["Region", f"Now<br><span style='font-weight:400;font-size:12px'>{issue.date()}</span>"] +
                       [f"{l} week{'s' if l > 1 else ''} ahead<br><span style='font-weight:400;font-size:12px'>{targets[l]}</span>" for l in leads])
        rows = ""
        for region, g in latest.groupby("region"):
            g = g.set_index("lead_weeks")
            cells = [f"<td style='padding:6px'>{region}</td>", f"<td style='padding:4px'>{level_cell(g['level_now'].iloc[0])}</td>"]
            cells += [f"<td style='padding:4px'>{level_cell(g.loc[l, 'forecast_level'], float(g.loc[l, 'p_d1plus']))}</td>" for l in leads]
            rows += "<tr>" + "".join(cells) + "</tr>"
        st.markdown(f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%'><tr>{head}</tr>{rows}</table></div>",
                    unsafe_allow_html=True)
        if len(warn):
            st.error("New-drought warnings: " + ", ".join(f"{r.region} ({r.lead_weeks} wk, {float(r.p_d1plus):.0%})" for r in warn.itertuples()))

        st.subheader("Chance of moderate drought (D1) or worse")
        ch = latest.assign(lead=latest["lead_weeks"].astype(str) + " wk", p=latest["p_d1plus"].astype(float))
        bars = alt.Chart(ch).mark_bar().encode(
            y=alt.Y("region:N", title=None, sort="-x"), x=alt.X("p:Q", title="Chance of D1 or worse", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            yOffset="lead:N", color=alt.Color("lead:N", title="Weeks ahead", scale=alt.Scale(scheme="oranges")),
            tooltip=["region", "lead", alt.Tooltip("p:Q", format=".0%", title="Chance of D1+"), "forecast_level"])
        rule = alt.Chart(pd.DataFrame({"x": [ALARM]})).mark_rule(strokeDash=[4, 4], color="#888").encode(x="x:Q")
        st.altair_chart(bars + rule, use_container_width=True)
        st.caption(f"Dashed line: the {ALARM:.0%} warning level. In the 2010-2025 test, a warning caught about half of new droughts 1-2 weeks ahead.")
        with st.expander("All probabilities"):
            cols = ["region", "lead_weeks", "target_date", "level_now", "forecast_level", "p_None", "p_D0", "p_D1", "p_D2+", "p_d1plus"]
            st.dataframe(latest[[c for c in cols if c in latest]], hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------- past
with tab_past:
    P = past_forecasts()
    if P.empty:
        st.info("No past forecasts yet. Run `python -m drought_rf.usdm ...` for the 2010-2025 test (HOW_TO_USE.md step 4b).")
    else:
        regions = sorted(P["region"].unique())
        a, b, c = st.columns([2, 1, 2])
        region = a.selectbox("Region", ["All regions"] + regions, index=0)
        lead = b.radio("Weeks ahead", sorted(P["lead_weeks"].unique()), index=1 if P["lead_weeks"].nunique() > 1 else 0, horizontal=True)
        y0, y1 = int(P["target_date"].dt.year.min()), int(P["target_date"].dt.year.max())
        yr = c.slider("Years", y0, y1, (max(y0, y1 - 4), y1)) if y1 > y0 else (y0, y1)
        sources = st.multiselect("Forecasts", ["Backtest", "Live"], default=sorted(P["source"].unique()),
                                 help="Backtest: made for 2010-2025 using only earlier years. Live: logged each week before the answer was known.")
        Q = P[(P["lead_weeks"] == lead) & P["target_date"].dt.year.between(*yr) & P["source"].isin(sources)]
        if region != "All regions":
            Q = Q[Q["region"] == region]
        done = Q.dropna(subset=["actual", "forecast"])
        if done.empty:
            st.info("None of these forecasts has reached its target week yet.")
        else:
            y, f, now = done["actual"].astype(int), done["forecast"].astype(int), done["level_now"].astype(int)
            lm, bm, pm = level_metrics(y, f), binary_metrics(y >= 2, done["p_d1plus"], ALARM), level_metrics(y, now)
            quiet = now < 2
            onset = quiet & (y >= 2)
            m = st.columns(6)
            m[0].metric("Exact level right", f"{lm['exact_level']:.0%}", f"{(lm['exact_level'] - pm['exact_level']) * 100:+.0f} pts vs copying")
            m[1].metric("MAE (levels)", f"{lm['mean_level_error']:.2f}", f"{lm['mean_level_error'] - pm['mean_level_error']:+.2f} vs copying", delta_color="inverse")
            m[2].metric("RMSE (levels)", f"{lm['rmse_levels']:.2f}", f"{lm['rmse_levels'] - pm['rmse_levels']:+.2f} vs copying", delta_color="inverse")
            m[3].metric("D1+ recall", "n/a" if np.isnan(bm["recall"]) else f"{bm['recall']:.0%}", help="Share of D1-or-worse weeks the forecast called")
            m[4].metric("D1+ precision", "n/a" if np.isnan(bm["precision"]) else f"{bm['precision']:.0%}", help="Share of D1+ calls that were right")
            m[5].metric("New droughts", f"{int((onset & (done['p_d1plus'] >= ALARM)).sum())} of {int(onset.sum())}", help="New droughts that got a warning, out of all new droughts")
            st.caption(f"{len(done):,} forecasts, {yr[0]}-{yr[1]}, {lead} week{'s' if lead > 1 else ''} ahead, "
                       f"{region.lower() if region == 'All regions' else region}.")

            show = region if region != "All regions" else st.selectbox("Region for the chart", regions)
            g = done[done["region"] == show].sort_values("target_date")
            if len(g):
                base = alt.Chart(g).encode(x=alt.X("target_date:T", title=None))
                lvl_axis = alt.Axis(values=[0, 1, 2, 3], labelExpr="['None','D0','D1','D2+'][datum.value]")
                actual = base.mark_line(interpolate="step-after", color="#333", strokeWidth=2).encode(
                    y=alt.Y("actual:Q", title="Level", axis=lvl_axis, scale=alt.Scale(domain=[-0.2, 3.2])))
                fc = base.mark_circle(size=28, opacity=.7).encode(
                    y="forecast:Q", color=alt.condition("datum.forecast == datum.actual", alt.value("#2a8a4a"), alt.value("#d1495b")),
                    tooltip=[alt.Tooltip("issue_date:T", title="Made from map"), alt.Tooltip("target_date:T", title="For week"),
                             alt.Tooltip("forecast:Q", title="Forecast level"), alt.Tooltip("actual:Q", title="Actual level"),
                             alt.Tooltip("p_d1plus:Q", format=".0%", title="Chance of D1+"), "source"])
                st.markdown(f"**{show}: actual level (line) and the forecast made {lead} week{'s' if lead > 1 else ''} earlier (dots: green right, red wrong)**")
                st.altair_chart((actual + fc).properties(height=230), use_container_width=True)
                prob = base.mark_area(opacity=.35, color="#e08a3c", interpolate="step-after").encode(
                    y=alt.Y("p_d1plus:Q", title="Chance of D1+", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])))
                ticks = alt.Chart(g[g["actual"] >= 2]).mark_tick(color="#8a2d0c", thickness=2, size=10).encode(x="target_date:T", y=alt.value(4))
                alarm = alt.Chart(pd.DataFrame({"y": [ALARM]})).mark_rule(strokeDash=[4, 4], color="#888").encode(y="y:Q")
                st.markdown("**Forecast chance of D1 or worse** (marks along the top: weeks that really were D1+)")
                st.altair_chart((prob + alarm + ticks).properties(height=150), use_container_width=True)

            st.markdown("**Each forecast**")
            tbl = done.sort_values("target_date", ascending=False).assign(
                **{"Made from map": done["issue_date"].dt.date, "For week": done["target_date"].dt.date,
                   "Level then": done["level_now"].astype(int).map(dict(enumerate(LEVELS))),
                   "Forecast": done["forecast"].astype(int).map(dict(enumerate(LEVELS))),
                   "Actual": done["actual"].astype(int).map(dict(enumerate(LEVELS))),
                   "Chance of D1+": (done["p_d1plus"] * 100).round(0).astype(int).astype(str) + "%",
                   "Right?": np.where(done["forecast"] == done["actual"], "✓", "✗"),
                   "Source": done["source"], "Region": done["region"]})
            st.dataframe(tbl[["Source", "Region", "Made from map", "For week", "Level then", "Forecast", "Actual", "Chance of D1+", "Right?"]],
                         hide_index=True, use_container_width=True, height=320)

# ----------------------------------------------------------------------------- live record
with tab_live:
    st.write("Every weekly forecast is saved in `forecasts/log.csv` before its answer is known, and never changed. "
             "Once the Drought Monitor maps the target week, it is scored here.")
    vs = read(FC / "verification_summary.csv")
    log = read(FC / "log.csv", keep_default_na=False, na_values=[""])
    if vs is not None and len(vs):
        v = vs.rename(columns={"lead_weeks": "Weeks ahead", "forecasts_checked": "Forecasts checked", "issue_weeks": "Weeks issued",
                               "exact_level": "Exact level", "persistence_exact": "Copying this week", "new_droughts": "New droughts",
                               "new_droughts_warned": "Warned"})
        for col in ("Exact level", "Copying this week"):
            v[col] = (v[col] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(v[[c for c in ["Weeks ahead", "Forecasts checked", "Weeks issued", "Exact level", "Copying this week", "New droughts", "Warned"] if c in v]],
                     hide_index=True, use_container_width=True)
    else:
        st.info("No live forecast has reached its target week yet. The first scores appear a week after the first logged forecast.")
    if log is not None and len(log):
        st.markdown(f"**Forecast log** ({log['issue_date'].nunique()} weeks issued)")
        st.dataframe(log.sort_values(["issue_date", "region", "lead_weeks"], ascending=[False, True, True]), hide_index=True, use_container_width=True)
