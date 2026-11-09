"""Streamflow drought labels: daily flow below the 20th variable-threshold percentile."""
from __future__ import annotations

import pandas as pd

from .percentiles import variable_threshold_percentile


def streamflow_percentiles(
    flow: pd.DataFrame, window: int = 30, ref_years: tuple[int, int] | None = None
) -> pd.DataFrame:
    """flow: long table with columns site, date, q. Returns site, date, q, q_pct.

    Zero-flow handling from the paper (threshold-level + continuous-dry method) is not
    implemented: Connecticut gauges in GAGES-II rarely reach zero flow. Add it if a
    chosen gauge does.
    """
    out = []
    for site, g in flow.groupby("site"):
        s = g.set_index("date")["q"].sort_index().asfreq("D")
        mask = None
        if ref_years:
            mask = pd.Series((s.index.year >= ref_years[0]) & (s.index.year <= ref_years[1]), index=s.index)
        pct = variable_threshold_percentile(s, window=window, ref_mask=mask)
        out.append(pd.DataFrame({"site": site, "date": s.index, "q": s.values, "q_pct": pct.values}))
    return pd.concat(out, ignore_index=True)


def drought_label(q_pct: pd.Series, threshold: float = 0.20) -> pd.Series:
    """1 = drought (percentile below threshold), 0 = not, NaN where flow is missing."""
    lab = (q_pct < threshold).astype("float")
    lab[q_pct.isna()] = float("nan")
    return lab
