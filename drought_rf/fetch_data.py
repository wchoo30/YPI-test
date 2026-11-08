"""Download the free inputs into data/raw/. Run this on your own machine: the cloud
container these files were written in cannot reach USGS, gridMET or NOAA, so this script
has NOT been run yet. Expect to fix small things the first time.

    pip install "dataretrieval>=1.4" pandas requests
    python fetch_data.py --sites 01188000,01193500 --start 1979-01-01 --end 2025-12-31

What it gets
  streamflow.csv       USGS daily discharge (00060) per gauge
  met.csv              gridMET daily prcp, tmin, tmax, pet at each gauge's location
                       (the paper averaged over the whole upstream basin; point values
                        at the gauge are a first pass, see README)
  teleconnections.csv  AMO, PDO, Nino 3.4, PNA monthly from NOAA PSL
Soil moisture and snow are NOT fetched here; see README "Data I need from you".
"""
import argparse
import io
from pathlib import Path

import pandas as pd
import requests

GRIDMET = {  # file stem -> (netCDF variable, output column, unit conversion)
    "pr": ("precipitation_amount", "prcp", lambda x: x),
    "tmmn": ("air_temperature", "tmin", lambda x: x - 273.15),
    "tmmx": ("air_temperature", "tmax", lambda x: x - 273.15),
    "pet": ("potential_evapotranspiration", "pet", lambda x: x),
}
NCSS = "http://thredds.northwestknowledge.net:8080/thredds/ncss/agg_met_{stem}_1979_CurrentYear_CONUS.nc"
PSL = {"amo": "https://psl.noaa.gov/data/correlation/amon.us.long.data",
       "pdo": "https://psl.noaa.gov/data/correlation/pdo.data",
       "nino34": "https://psl.noaa.gov/data/correlation/nina34.anom.data",
       "pna": "https://psl.noaa.gov/data/correlation/pna.data"}


def usgs(sites, start, end):
    """Daily discharge. Uses the new USGS Water Data API (dataretrieval >= 1.4,
    waterdata module); falls back to the legacy NWIS service, which USGS retires in
    early 2027."""
    try:
        from dataretrieval import waterdata
        df, _ = waterdata.get_daily(monitoring_location_id=[f"USGS-{s}" for s in sites],
                                    parameter_code="00060", time=f"{start}/{end}")
        tcol = "time" if "time" in df else [c for c in df.columns if "date" in c or "time" in c][0]
        out = pd.DataFrame({"site": df["monitoring_location_id"].str.replace("USGS-", "", regex=False),
                            "date": pd.to_datetime(df[tcol]).dt.tz_localize(None).dt.normalize(),
                            "q": pd.to_numeric(df["value"], errors="coerce")})
    except (ImportError, AttributeError) as e:
        print(f"waterdata module unavailable ({e}); using legacy nwis")
        from dataretrieval import nwis
        df, _ = nwis.get_dv(sites=sites, parameterCd="00060", start=start, end=end)
        df = df.reset_index()
        qcol = [c for c in df.columns if c.startswith("00060") and not c.endswith("_cd")][0]
        out = pd.DataFrame({"site": df["site_no"], "date": pd.to_datetime(df["datetime"]).dt.tz_localize(None).dt.normalize(),
                            "q": pd.to_numeric(df[qcol], errors="coerce")})
    from dataretrieval import nwis  # site metadata (lat/lon, area); still on the legacy service
    info, _ = nwis.get_info(sites=sites)
    locs = info[["site_no", "dec_lat_va", "dec_long_va", "drain_area_va"]].rename(
        columns={"site_no": "site", "dec_lat_va": "lat", "dec_long_va": "lon", "drain_area_va": "drain_area_mi2"})
    return out, locs


def gridmet_point(lat, lon, start, end):
    cols = {}
    for stem, (var, col, conv) in GRIDMET.items():
        r = requests.get(NCSS.format(stem=stem), params={
            "var": var, "latitude": lat, "longitude": lon, "accept": "csv",
            "time_start": f"{start}T00:00:00Z", "time_end": f"{end}T00:00:00Z"}, timeout=300)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text))
        d["date"] = pd.to_datetime(d.iloc[:, 0]).dt.tz_localize(None).dt.normalize()
        cols[col] = pd.Series(conv(d.iloc[:, -1].astype(float)).values, index=d["date"])
    return pd.DataFrame(cols).rename_axis("date").reset_index()


def psl_monthly(url):
    lines = requests.get(url, timeout=120).text.splitlines()
    y0, y1 = (int(v) for v in lines[0].split()[:2])
    rows = []
    for ln in lines[1:]:
        p = ln.split()
        if len(p) == 13 and p[0].isdigit() and y0 <= int(p[0]) <= y1:
            for m, v in enumerate(p[1:], 1):
                v = float(v)
                rows.append((pd.Timestamp(int(p[0]), m, 1), None if v <= -99 else v))
    return pd.Series(dict(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", required=True, help="USGS gauge numbers, comma separated")
    ap.add_argument("--start", default="1979-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", default="data/raw")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sites = a.sites.split(",")

    flow, locs = usgs(sites, a.start, a.end)
    flow.to_csv(out / "streamflow.csv", index=False)
    locs.to_csv(out / "gauges.csv", index=False)
    print(f"streamflow: {len(flow)} rows")

    met = []
    for _, r in locs.iterrows():
        m = gridmet_point(r["lat"], r["lon"], a.start, a.end)
        m.insert(0, "site", r["site"])
        met.append(m)
        print(f"gridMET {r['site']}: {len(m)} days")
    pd.concat(met).to_csv(out / "met.csv", index=False)

    tele = pd.DataFrame({k: psl_monthly(u) for k, u in PSL.items()}).rename_axis("date").reset_index()
    tele.to_csv(out / "teleconnections.csv", index=False)
    print("teleconnections:", tele["date"].min().date(), "to", tele["date"].max().date())
