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
import re
import time
from pathlib import Path

import pandas as pd
import requests

GRIDMET = {  # file stem -> (expected netCDF variable, output column, unit conversion)
    "pr": ("precipitation_amount", "prcp", lambda x: x),
    "tmmn": ("daily_minimum_temperature", "tmin", lambda x: x - 273.15),
    "tmmx": ("daily_maximum_temperature", "tmax", lambda x: x - 273.15),
    "pet": ("daily_mean_reference_evapotranspiration_grass", "pet", lambda x: x),
}
NCSS = "http://thredds.northwestknowledge.net:8080/thredds/ncss/agg_met_{stem}_1979_CurrentYear_CONUS.nc"
def _day(x):
    """Any date/time column (naive or with a time zone) -> naive calendar day."""
    return pd.to_datetime(x, utc=True).dt.tz_convert(None).dt.normalize()


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
                            "date": _day(df[tcol]),
                            "q": pd.to_numeric(df["value"], errors="coerce")})
    except Exception as e:  # old dataretrieval, or the new API refused the request
        print(f"waterdata module unavailable ({e}); using legacy nwis")
        from dataretrieval import nwis
        df, _ = nwis.get_dv(sites=sites, parameterCd="00060", start=start, end=end)
        df = df.reset_index()
        qcol = [c for c in df.columns if c.startswith("00060") and not c.endswith("_cd")][0]
        out = pd.DataFrame({"site": df["site_no"], "date": _day(df["datetime"]),
                            "q": pd.to_numeric(df[qcol], errors="coerce")})
    from dataretrieval import nwis  # site metadata (lat/lon, area); still on the legacy service
    info, _ = nwis.get_info(sites=sites)
    locs = info[["site_no", "dec_lat_va", "dec_long_va", "drain_area_va"]].rename(
        columns={"site_no": "site", "dec_lat_va": "lat", "dec_long_va": "lon", "drain_area_va": "drain_area_mi2"})
    return out, locs


def _ncss_var(stem, expected):
    """Ask the server what the file's data variable is really called (names differ by file)."""
    try:
        txt = requests.get(NCSS.format(stem=stem) + "/dataset.xml", timeout=60).text
        names = re.findall(r'<grid name="([^"]+)"', txt)
    except requests.RequestException:
        names = []
    if expected in names or not names:
        return expected
    return names[0]


def _get_with_retries(url, params, tries=5, timeout=300):
    """gridMET's server sometimes drops a connection mid-download; wait and try again."""
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code >= 500:
                raise requests.HTTPError(f"server error {r.status_code}", response=r)
            r.raise_for_status()
            return r.text
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
                requests.HTTPError) as e:
            bad_request = isinstance(e, requests.HTTPError) and e.response is not None and 400 <= e.response.status_code < 500
            if bad_request or attempt == tries:
                raise
            wait = 5 * 2 ** (attempt - 1)
            print(f"    connection problem ({type(e).__name__}), retrying in {wait}s ({attempt}/{tries - 1})", flush=True)
            time.sleep(wait)


def gridmet_point(lat, lon, start, end, cache_dir=None, chunk_years=5):
    """Daily gridMET values at one point. Downloads in `chunk_years` pieces with retries,
    and saves each piece in cache_dir so a rerun skips what is already done."""
    y0, y1 = int(start[:4]), int(end[:4])
    cols = {}
    for stem, (expected, col, conv) in GRIDMET.items():
        var = _ncss_var(stem, expected)
        parts = []
        for cy in range(y0, y1 + 1, chunk_years):
            c_start = max(start, f"{cy}-01-01")
            c_end = min(end, f"{min(cy + chunk_years - 1, y1)}-12-31")
            cache = Path(cache_dir) / f"{stem}_{lat:.4f}_{lon:.4f}_{c_start}_{c_end}.csv" if cache_dir else None
            if cache is not None and cache.exists():
                text = cache.read_text()
            else:
                print(f"  {col}: {c_start[:4]}-{c_end[:4]}", flush=True)
                text = _get_with_retries(NCSS.format(stem=stem), {
                    "var": var, "latitude": lat, "longitude": lon, "accept": "csv",
                    "time_start": f"{c_start}T00:00:00Z", "time_end": f"{c_end}T00:00:00Z"})
                if cache is not None:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(text)
            d = pd.read_csv(io.StringIO(text))
            # columns look like: time, latitude[...], longitude[...], precipitation_amount[unit="mm"]
            vcol = next((c for c in d.columns if c.startswith(var)), d.columns[-1])
            parts.append(pd.Series(conv(pd.to_numeric(d[vcol], errors="coerce")).values,
                                   index=_day(d[d.columns[0]]).values))
        sr = pd.concat(parts)
        cols[col] = sr[~sr.index.duplicated()]
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
    ap.add_argument("--skip-streamflow", action="store_true", help="reuse streamflow.csv and gauges.csv already in --out")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sites = a.sites.split(",")

    if a.skip_streamflow:
        locs = pd.read_csv(out / "gauges.csv", dtype={"site": str})
        print("streamflow: reusing", out / "streamflow.csv")
    else:
        flow, locs = usgs(sites, a.start, a.end)
        flow.to_csv(out / "streamflow.csv", index=False)
        locs.to_csv(out / "gauges.csv", index=False)
        print(f"streamflow: {len(flow)} rows")

    met = []
    for _, r in locs.iterrows():
        print(f"gridMET {r['site']}:", flush=True)
        m = gridmet_point(r["lat"], r["lon"], a.start, a.end, cache_dir=out / "_gridmet_cache")
        m.insert(0, "site", r["site"])
        met.append(m)
        print(f"gridMET {r['site']}: {len(m)} days")
    pd.concat(met).to_csv(out / "met.csv", index=False)

    tele = pd.DataFrame({k: psl_monthly(u) for k, u in PSL.items()}).rename_axis("date").reset_index()
    tele.to_csv(out / "teleconnections.csv", index=False)
    print("teleconnections:", tele["date"].min().date(), "to", tele["date"].max().date())
