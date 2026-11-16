# Connecticut drought indicators: what each dataset gives, and its role in the prototype

Built from Winston's list (Connecticut Water Planning Council drought factors), checked
2026-11-08. "Role" is what the dataset does in the first real-data prototype.

| Dataset | URL | Access type | Location granularity | Time granularity / range | Verified / status | Role in prototype |
|---|---|---|---|---|---|---|
| **USGS streamflow** (Water Data for the Nation) | https://waterdata.usgs.gov/state/Connecticut/ | Free API (`dataretrieval.waterdata`); old WaterServices API retires Q1 2027 | Point, one per gauge (~100 CT gauges, about a dozen long unregulated records) | Daily mean discharge; most long gauges 1930s–present; percentiles via statistics service | Verified (USGS blog, dataretrieval 1.4 on PyPI) | **Target**: drought = flow below 20th percentile for the time of year (as in the HESS paper) |
| **PRISM** | https://prism.oregonstate.edu/ | Free download: gridded files, web service, FTP; Explorer gives point time series, up to 500 points per request | 800 m and 4 km grid | Daily, 1981–present; also monthly and 30-yr normals | Verified on PRISM homepage: daily 800 m and 4 km free from 1981 | **Predictor**: precipitation, Tmin, Tmax, VPD; 30/90/365-day totals and percentiles. Best choice for CT's local rainfall differences |
| readnoaa (R package) | https://ftp.belnet.be/mirror/CRAN/web/packages/readnoaa/readme/README.html | R wrapper for NOAA station data | Point, per weather station | Daily, station record | Not checked in detail | Not needed if PRISM is used. GHCN-Daily stations (same data) are a backup for checking PRISM |
| **USGS groundwater levels** (NGWMN / NWIS) | https://www.usgs.gov/apps/ngwmn/ | Free API | Point, per well. Sparse: many wells measured monthly by hand, a few continuous | Daily (continuous wells) or monthly (manual); varies per well | Gaps confirmed as you noted | **Predictor**: level percentile at the nearest long-record well. Use only wells with continuous records; slow groundwater adds skill at 30 days |
| Drinking water reservoir levels (CT DPH) | https://portal.ct.gov/dph/drinking-water/dws/reservoir-data | PDF reports only | Per public water system, with county and statewide averages | Roughly weekly, irregular dates; each report is one snapshot | As you described | **Leave out of prototype**: no time series without asking DPH for the history. Worth requesting for the live version |
| **Palmer Drought Severity Index** (NOAA nClimDiv) | https://www.ncei.noaa.gov/pub/data/cirs/climdiv/ | Free text/CSV files | 3 CT climate divisions | Monthly, 1895–present | Verified earlier (data-sources.md) | **Predictor**, lagged one month (published after month end); also a benchmark |
| Crop Moisture Index (CPC) | https://cpc.ncep.noaa.gov/products/analysis_monitoring/cdus/palmer_drought/index.shtml | Weekly tables | 3 CT climate divisions | Weekly, 1973 to ~2023; discontinued | Discontinued | **Leave out**: it stops, so a live model can't use it |
| Vegetation Drought Response Index (USGS EROS) | https://www.usgs.gov/centers/eros/science/usgs-eros-archive-vegetation-monitoring-vegetation-drought-response-index | Free gridded download | 1 km grid | Weekly, 2009–present | As you described; not re-checked | **Optional predictor** later. Only 2009 onward, which halves the training record |
| Fire danger (CT DEEP Forestry) | DEEP daily report | Daily web report | Statewide / regional | Daily; no archive found | Not checked | **Leave out**: no history to train on. Mostly reflects weather we already have |
| **U.S. Drought Monitor** | https://droughtmonitor.unl.edu/ | Free API and CSV (county statistics) | State and county | Weekly (Thursday maps), 2000–present | Verified earlier | **Benchmark and second target**: does our forecast lead the USDM category? Not a predictor, since USDM authors already use streamflow and PDSI |
| Coastal water temperature (NOAA NDBC) | https://www.ndbc.noaa.gov/ | Free station files | Point, buoys on the Sound | 10–60 min, daily summaries | Not checked | **Leave out**: not a driver of inland drought |

## Prototype in one line

Predict whether each Connecticut stream gauge will be in drought 0, 7, 14 and 30 days
from now. Predictors: PRISM weather averaged over the gauge's basin, the nearest
groundwater well, Palmer index, and climate indices. Compare against persistence,
climatology and the Drought Monitor.

## How this leads to the live farm model

Everything in the prototype updates daily to monthly, so the same pipeline can run live.
Farm sensors join later in two ways. Soil moisture probes become a farm-level
target (is this field dry), which the stream-level model cannot give. Their readings
also become predictors for nearby cells. The prototype's job is only to show that real
public data predicts drought ahead of time better than the baselines.
