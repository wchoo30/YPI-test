"""Download NOAA nClimGrid-Daily county averages (precip, Tmax, Tmin) for Connecticut.

Source: AWS open data bucket noaa-nclimgrid-daily-pds (EpiNOAA county parquet files).
Usage (from the drought_rf folder):  python scripts/fetch_nclimgrid_counties.py data/raw/ct_county_daily.csv
Needs pandas and pyarrow. Takes ~20 s. The last days of the newest month can hold
negative fill values; build_dashboard_data.py drops them.
"""
import io, re, sys, concurrent.futures as cf, urllib.request
import pandas as pd
B = "https://noaa-nclimgrid-daily-pds.s3.amazonaws.com"
def keys(year):
    t = urllib.request.urlopen(f"{B}/?list-type=2&prefix=EpiNOAA/v1-0-0/parquet/cty/YEAR={year}/").read().decode()
    ks = re.findall(r"<Key>(.*?)</Key>", t)
    # prefer scaled over prelim for the same month
    by = {}
    for k in ks:
        m = k.rsplit("/", 1)[1][:6]
        if m not in by or "scaled" in k:
            by[m] = k
    return list(by.values())
def get(k):
    d = pd.read_parquet(io.BytesIO(urllib.request.urlopen(f"{B}/{k}").read()))
    d = d[d["fips"].astype(str).str.startswith("09")]
    return d[["fips", "region_name", "date", "tmax", "tmin", "tavg", "prcp"]]
allk = [k for y in range(1990, pd.Timestamp.today().year + 1) for k in keys(y)]
print(len(allk), "files", flush=True)
with cf.ThreadPoolExecutor(16) as ex:
    parts = list(ex.map(get, allk))
df = pd.concat(parts)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["fips", "date"]).drop_duplicates(["fips", "date"], keep="last")
df.to_csv(sys.argv[1], index=False)
print(df.shape, df["date"].min(), df["date"].max(), df["region_name"].unique())
