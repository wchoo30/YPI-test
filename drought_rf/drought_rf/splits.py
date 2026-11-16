"""Train/test splits.

paper_split: train 1985-2015, test 1982-1984 and 2016-2020 (Heldmyer et al. 2026).
walk_forward: ctwater's convention. Train on all years before a test year, drop the
`embargo_days` before the test year from training so the label window cannot leak.
"""
from __future__ import annotations

import pandas as pd


def paper_split(df: pd.DataFrame, date_col: str = "date"):
    y = df[date_col].dt.year
    train = (y >= 1985) & (y <= 2015)
    test = ((y >= 1982) & (y <= 1984)) | ((y >= 2016) & (y <= 2020))
    yield "paper", train.values, test.values


def walk_forward(df: pd.DataFrame, first_test_year: int, last_test_year: int,
                 embargo_days: int = 30, min_train_years: int = 10, date_col: str = "date"):
    d = df[date_col]
    target = df["target_date"] if "target_date" in df else d
    start = d.min().year
    for yr in range(first_test_year, last_test_year + 1):
        if yr - start < min_train_years:
            continue
        cut = pd.Timestamp(f"{yr}-01-01")
        train = (target < cut - pd.Timedelta(days=embargo_days)).values
        test = (d.dt.year == yr).values
        if test.any():
            yield str(yr), train, test
