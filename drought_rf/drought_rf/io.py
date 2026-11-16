"""Input files. Put them in one folder (default data/raw/):

streamflow.csv       site, date, q                     daily mean discharge (cfs), USGS 00060
met.csv              site, date, prcp, tmin, tmax, pet, sm_shallow, sm_deep, swe
                     (any subset of the weather columns works; units: mm, degC, mm, 0-1 or mm)
teleconnections.csv  date (first of month), amo, pdo, nino34, pna      optional
static.csv           site, <basin attributes ...>                      optional, donor step only
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read(path: Path, required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    df = pd.read_csv(path, dtype={"site": str})
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load(data_dir: str | Path):
    d = Path(data_dir)
    return (_read(d / "met.csv"), _read(d / "streamflow.csv"),
            _read(d / "teleconnections.csv", False), _read(d / "static.csv", False))
