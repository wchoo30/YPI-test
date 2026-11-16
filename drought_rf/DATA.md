# Data catalog for drought_rf

Checked 2026-11-02 against each source's own page (links below). "Reachable here"
means the cloud container that builds this code can download it; everything else has
to be fetched on your machine with `fetch_data.py` or by hand.

## The unit everything has to line up on

The model works on **one row per gauge basin per day**. The label is a gauge's daily
flow, so every predictor has to be turned into a daily value for that gauge's upstream
basin (area-weighted average of the grid cells inside the basin polygon). The paper did
exactly this with the `gdptools` package. Farm cells (1 km) join later through the donor
step, which needs the same predictors averaged over the farm's own small catchment.

| Thing | Native grain | What we turn it into |
|---|---|---|
| Label (streamflow) | daily, one point per gauge | daily percentile per gauge (done in code) |
| Weather, soil moisture, snow | gridded, hourly to daily | daily, averaged over each basin polygon |
| Climate indices | monthly, one number for the globe | forward-filled to daily, lagged 1 month |
| Basin attributes | one value per basin, static | used only by the donor step |
| Farm cells | 1 km cell | needs its own catchment polygon for the donor step |

## Inputs, source by source

| Input | Paper used | Grid | Time step | Period | Access | Reachable here |
|---|---|---|---|---|---|---|
| Streamflow (label) | USGS daily values, GAGES-II gauges | point | daily | gauge record | USGS Water Data API (`dataretrieval.waterdata.get_daily`) | no |
| Precip, Tmin, Tmax, PET | gridMET | ~4 km | daily | 1979 to ~4 days ago | climatologylab.org / THREDDS / Climate Engine | no |
| SPEI | gridMET drought product (or computed from P − PET) | ~4 km | 5-day / pentad | 1979 to present | same | no |
| Soil moisture 10–40, 40–100 cm | NLDAS-2 Noah, basin-averaged by USGS | 12.5 km | daily (from hourly) | **1980–2020 only** | USGS release doi:10.5066/P9IKHEM3, 7 CSVs, 0.6–0.9 GB each | no |
| Soil moisture, after 2020 | (not in paper) NLDAS-2 Noah direct | 12.5 km | hourly | 1979 to ~4 days ago | NASA GES DISC, free Earthdata login | no |
| Snow water equivalent | UA SWE (Broxton et al.) | 4 km | daily | **Oct 1981 – Sep 2023 only** | NSIDC-0719 | no |
| AMO, PDO, Niño 3.4, PNA | NOAA indices | none | monthly | 1948+ | NOAA PSL text files | no |
| Basin attributes | GAGES-II (58 variables) | per basin | static | 2011 vintage | USGS GAGES-II | no |
| Gauge list | 3,198 gauges passing a record-completeness test | | | | in the paper's release, doi:10.5066/P1FSRPMU | no |

### Finer options that ctwater already uses or that I can download here

| Input | Grid | Time step | Period | Notes |
|---|---|---|---|---|
| AORC weather (ctwater's forcing) | ~800 m | hourly | 1979–2025 | On AWS (`noaa-nws-aorc-v1-1-1km`), **reachable here**. Pulling Connecticut for all years is tens of GB, so better to export ctwater's existing copy. |
| National Water Model soil moisture (4 layers: 0–10, 10–40, 40–100, 100–200 cm) | 1 km | 3-hourly | Feb 1979 – Jan 2023 (v3.0 retrospective) | On AWS (`noaa-nwm-retrospective-3-0-pds`), **reachable here**. About 7 GB of downloads for Connecticut's two middle layers. Same 1 km grid as farm cells. After Jan 2023 the operational NWM uses different forcing, so the series has a seam. |
| NLDAS-3 | 1 km | hourly | open-loop run from ~1980; near-real-time ~7 h latency | NASA's successor to NLDAS-2. Rollout was planned for 2025–26; not yet checked whether it is downloadable. |

## Where the periods overlap

```
                      1979  1982        2000          2020 2023   2026
USGS flow (label)      ├───────────────────────────────────────────────┤
gridMET / AORC         ├───────────────────────────────────────────────┤
NLDAS-2 basin release    ├──────────────────────────────┤
NLDAS-2 direct           ├───────────────────────────────────────────────┤
UA SWE                      ├────────────────────────────────────┤
NWM v3 soil moisture   ├──────────────────────────────────────┤
Paper train/test             test ├──── train 1985–2015 ────┤ test ┤
```

* **To reproduce the paper** (1982–2020) everything above is available.
* **To forecast today**, two of the paper's inputs stop too early: the basin soil moisture
  release ends in 2020 and UA SWE ends in Sep 2023. The fix is to take soil moisture
  *and* snow from NLDAS-2 Noah directly (same model as the release, runs to ~4 days
  ago), so training and live inputs come from one source. Snow matters little in most
  of Connecticut.

## Granularity problems to know about

1. **Small basins vs coarse grids.** Several CT reference gauges drain only a few
   square miles (Bunnell Brook, 01188000, is one; check its area), smaller than one
   12.5 km NLDAS cell (about 150 km²). For basins that small, "basin-averaged soil
   moisture" is really one cell's value. gridMET at 4 km is fine for most CT basins.
   NWM or NLDAS-3 at 1 km would fix this, at the cost of the seam or the unknown
   availability above.
2. **Point vs basin.** `fetch_data.py` currently takes gridMET at the gauge's point
   location. That is a first pass. The proper version averages over the basin polygon
   (GAGES-II boundaries, or the USGS NLDI basin service), which I can add once the
   polygons are downloaded.
3. **Farms vs gauges.** The model predicts *streamflow* drought for a basin. A farm cell
   gets a prediction for its own small catchment through donor gauges. That still
   says "the local streams are in drought," not "this field's soil is dry." For the
   field, ctwater's water-balance target is the right label, and this model's output is
   a predictor for it.
4. **Monthly indices.** NOAA publishes AMO/PDO/etc. after the month ends, so the code
   lags them a month to avoid using numbers that would not exist yet on the forecast
   day.
5. **USGS API change.** USGS is shutting down `waterservices.usgs.gov` in early 2027
   and some outages may start this winter. `fetch_data.py` now uses the new
   `dataretrieval.waterdata` module (dataretrieval ≥ 1.4) and falls back to the old one.

## Sources checked

- Paper: https://hess.copernicus.org/articles/30/5925/2026/
- Paper's models and outputs: https://www.usgs.gov/data/random-forest-and-donor-based-simulated-streamflow-drought-3198-gages-across-conus-1982-2020
- NLDAS-2 Noah basin averages: https://usgs.gov/data/daily-average-soil-moisture-and-ancillary-data-noah-land-surface-model-national-land-data (doi:10.5066/P9IKHEM3)
- gridMET: https://www.drought.gov/data-maps-tools/gridded-surface-meteorological-gridmet-dataset
- UA SWE: https://nsidc.org/data/nsidc-0719
- NWM retrospective: https://registry.opendata.aws/nwm-archive/
- NLDAS-3 plans (Apr 2025 slides): https://ldas.gsfc.nasa.gov/sites/default/files/ldas/nldas/NLDAS-3_Drought-Workshop-3_Slides.pdf
- USGS WaterServices shutdown: https://waterdata.usgs.gov/blog/api-waterservices-decom/
- dataretrieval: https://pypi.org/project/Dataretrieval/
- USGS already runs a related forecast, River DroughtCast (streamflow drought up to 90 days ahead, weekly, ML): https://www.usgs.gov/tools/river-droughtcast — worth using as a benchmark for our 30-day forecasts.
