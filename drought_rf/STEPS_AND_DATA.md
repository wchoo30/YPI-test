# What each step does, and where the data comes from

This covers the commands in HOW_TO_USE.md for the **Drought Monitor level model** (the one
with real results). The stream gauge model is explained in WALKTHROUGH.md.

## The datasets

| Dataset | Who makes it | What it is | How it got here |
|---|---|---|---|
| U.S. Drought Monitor, CT planning regions | National Drought Mitigation Center (UNL), USDA, NOAA | Weekly percent of each of CT's 9 planning regions at each level (D0–D4 or worse), Jan 2000 – Oct 2026 | Already in the repo: `data/ct_drought_monitor_2000_2026.csv`, the export from the Drought Monitor's data page used by your earlier notebook. Nothing is downloaded for it. |
| NOAA nClimGrid-Daily, county averages | NOAA National Centers for Environmental Information | Daily rainfall, max and min temperature on a ~5 km grid, averaged by NOAA over each US county. CT has 8 (pre-2022) counties. 1951 to about 3 months ago | Downloaded by `scripts/fetch_nclimgrid_counties.py` from NOAA's public bucket on Amazon Web Services (AWS Open Data): `https://noaa-nclimgrid-daily-pds.s3.amazonaws.com/EpiNOAA/v1-0-0/parquet/cty/` |

Not used yet (the cloud setup I worked in can't reach their websites; `fetch_data.py`
gets them on your computer): USGS streamflow, gridMET/PRISM weather, soil moisture,
groundwater, climate indices.

### How the rainfall download works

NOAA stores one file per month, each holding every US county's daily values, in Parquet
format (a compressed table format). The script:

1. Asks the bucket for the list of files in each year folder, 1990 to this year
   (`?list-type=2&prefix=...` is the standard Amazon S3 listing request; no account needed).
2. For each month keeps the final version (`STATUS=scaled`) and falls back to the
   preliminary one (`STATUS=prelim`) for the newest months.
3. Downloads the ~430 files, 16 at a time (about 0.8 MB each, ~20 seconds in total).
4. Keeps only rows whose county code starts with `09` (Connecticut).
5. Saves one table: county, date, tmax, tmin, tavg, prcp → `data/raw/ct_county_daily.csv`.

The newest few days can carry negative "missing" codes; the model drops them.

## What each command does

### `pip install -r requirements.txt`
Installs the Python libraries: pandas and numpy (tables and math), scikit-learn (the
random forest), lightgbm (the comparison model), pyarrow (reads NOAA's Parquet files),
matplotlib and jupyter (the charts notebook), dataretrieval and requests (download scripts
for the gauge model), pytest (the tests).

### `python scripts/fetch_nclimgrid_counties.py data/raw/ct_county_daily.csv`
The rainfall download above.

### `python -m drought_rf.usdm --usdm ... --weather ...`
Runs `drought_rf/usdm.py`, in this order:

1. **Turn the Drought Monitor file into one level per region per week.** A region's level is
   the worst level covering at least half its area: None, D0, D1, or D2+. (D3 covered half
   a region in only 67 region-weeks and D4 never did, so they join D2.)
2. **Give each region its weather.** Each planning region takes the county holding most of
   its area (e.g. Capitol → Hartford County, Northwest Hills → Litchfield). Evaporation
   demand is estimated from temperature (Hargreaves formula), since NOAA's county files
   have no evaporation.
3. **Build the predictors** (`features.py`), about 100 per region per day:
   rainfall, temperature, evaporation demand and rainfall minus evaporation; their 30-,
   90- and 365-day totals or averages; each of those ranked against the same time of
   year in 1991–2020 (seasonal percentile) and against the whole record; the date.
4. **Add where the region stands now:** its current level, its level 4 weeks ago, its
   share of area in D1 or worse, and how much that share changed over 4 weeks.
5. **Line up the question.** Each row has predictors known on week *t* and the answer is
   the level on week *t* + 0, 1, 2 or 4 weeks.
6. **Train and test year by year (2010–2025).** To predict a year it trains only on
   earlier years, leaving a 4-week gap. For each year it trains two random forests
   (500 trees, at least 10 weeks per leaf, rarer levels weighted up): one with weather
   only, one with weather plus the current level. It also makes two simple baseline
   guesses: "same as this week" and "most common level for this region and month".
7. **Score and save** to `outputs/usdm/`:
   `predictions.csv` (every region-week: actual level, each model's predicted level and
   the forest's probability of each level), `summary.csv` (scores per model and horizon),
   `rf_importance.csv` (which predictors the forest used most).

### `results.ipynb` → Run All
Reads those three files and draws the charts: exact-level accuracy, new-drought warnings
vs false alarms, predicted vs actual over time for one region, a predicted-vs-actual
table, and the top predictors.
