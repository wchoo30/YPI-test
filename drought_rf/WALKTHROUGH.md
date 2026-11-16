# drought_rf walkthrough

What the code does, file by file, and exactly what goes into the YPI-test repo.

## 1. What will be pushed to GitHub, and how

**The repo today.** `wchoo30/YPI-test` is **public**. Its main branch is
`claude/csv-pandas-visualization-qawld2`. It holds `analysis.ipynb`, a Drought Monitor
CSV for CT's 9 planning regions (2000–2026), a README and a requirements.txt.

**What was done:**
1. Made a new branch, `drought-rf-prototype`, off the main branch. Nothing already in the
   repo is changed.
2. Added one new folder, `drought_rf/`, holding everything listed in section 2.
3. Added a `.gitignore` so downloaded data (`drought_rf/data/raw/`) and model outputs
   (`outputs/`) never get committed. Real data files are large and can be re-downloaded.
4. Added a short section to the repo README pointing to `drought_rf/`.
5. Opened a pull request into the main branch. You (and the group) review it on GitHub and
   merge it when happy. Until it is merged, the main branch stays exactly as it is.

**Not pushed:** any data, model outputs or passwords. The repo is public, so anything
in it is visible to anyone.

## 2. Files

```
drought_rf/
├── README.md            what the model is, the prediction diagram, models, how to run
├── DATA.md              every input's grid size, time step, years and download route
├── INDICATORS.md        the CT drought indicator table (role of each dataset)
├── WALKTHROUGH.md       this file
├── HOW_TO_USE.md        step-by-step commands
├── scripts/             nClimGrid rainfall download + chart data builder
├── requirements.txt     Python packages
├── fetch_data.py        downloads real data into data/raw/  (run on your own computer)
├── drought_rf/          the Python package
│   ├── percentiles.py   turns any daily series into "how unusual for this time of year"
│   ├── targets.py       streamflow → drought yes/no label
│   ├── features.py      builds the predictor table, and lines it up with future labels
│   ├── models.py        random forest, LightGBM, logistic, climatology, persistence
│   ├── splits.py        which years train and which years test
│   ├── evaluate.py      scores (kappa, hit rate, false alarm, Brier skill)
│   ├── donor.py         borrows gauge models for places with no gauge (farms)
│   ├── io.py            reads the CSV files in data/raw/
│   ├── synthetic.py     fake data, only for testing that the code runs
│   └── pipeline.py      runs everything start to finish
└── tests/               6 automated checks
```

## 3. How a prediction is made, step by step

**Step 1: label (targets.py, percentiles.py).** For each stream gauge, each day's flow is
ranked against every other year's flow within ±15 days of that date. A flow lower than
80 % of comparable days counts as drought (below the 20th percentile). Ranking within
the season matters: low flow in August is normal, the same flow in April is a drought.
The ranking uses training years only, so test years don't leak into their own thresholds.

**Step 2: predictors (features.py).** For each gauge and day:
- daily rainfall, temperature, evaporation demand, soil moisture, snow;
- their 30-, 90- and 365-day totals or averages (drought builds slowly);
- each of those ranked the same way as flow (seasonal percentile) and against the
  whole record;
- rainfall minus evaporation demand (a stand-in for the SPEI drought index);
- monthly climate indices (El Niño, PDO, AMO, PNA), shifted a month later because
  NOAA publishes them after the month ends;
- the date and the time of year.

Optionally (`--use-flow`), today's flow percentile too. The paper left it out so the
model could work where there is no gauge. For forecasting at a gauge it helps a lot.

**Step 3: line up with the future (features.py `make_supervised`).** A row with
predictors from day *t* is paired with the drought label from day *t + lead*. Lead 0 is
the paper's version (is there drought today). Leads 7, 14 and 30 are real forecasts,
since nothing after day *t* is in the row. A test checks this alignment.

**Step 4: train (models.py, splits.py).** One model per gauge per lead, as in the paper.
The random forest uses the paper's settings: 500 trees, at least 10 days per leaf,
drought days weighted 4× (0.8 vs 0.2) because drought is only 20 % of days. Two ways
to split years:
- `paper`: train 1985–2015, test 1982–84 and 2016–20, to compare with the paper;
- `walk_forward`: for each test year, train on all earlier years, with a 30-day gap so
  no label overlaps the test year. This is how a live model would actually be used.

**Step 5: score (evaluate.py).** For each gauge, lead and model:
- **kappa** (the paper's main score; 0 = no better than chance, 1 = perfect),
- **sensitivity** (share of drought days caught), **specificity** (share of normal
  days correctly left alone),
- **Brier skill** (how much better the probabilities are than "usual drought rate for
  this time of year"; above 0 means better).

The random forest is compared with LightGBM, logistic regression, climatology, and
persistence ("drought today, so drought in 30 days"). A forecast is only useful if it
beats persistence.

**Step 6: farms (donor.py).** A farm has no gauge. The paper's fix: record which
predictors each gauge's forest relies on, learn how that pattern relates to basin traits
(soil, land cover, size, elevation), then for a farm pick the gauges whose pattern
should look most like it and average their forecasts. Not run yet; it needs the real
gauges first.

**Outputs** (in `outputs/`): `predictions.csv` (every test day, every model),
`scores.csv`, `summary.csv`, `rf_importance.csv` (which predictors mattered).

## 4. What has been tested

- 6 tests pass: percentiles come out uniform, seasonality is removed, the lead
  alignment is right, the walk-forward gap holds, the full pipeline runs, and donor
  selection picks the right gauge on a toy example.
- The full pipeline runs on **synthetic data only**. No real data has gone through it,
  and the demo's scores are not results.
- `fetch_data.py` has **never been run**, because the cloud setup can't reach USGS,
  gridMET or PRISM. Expect small fixes the first time you run it.

## 5. Next steps after the merge

1. Switch weather from gridMET to PRISM (your preferred source) and add groundwater wells
   and the Palmer index loaders, per INDICATORS.md.
2. Run `fetch_data.py` for 8–10 CT gauges (in VS Code on your computer, or here once the
   network domains are allowed).
3. Run the real prototype with `--split walk_forward --leads 0,7,14,30`.
4. Use the repo's existing Drought Monitor CSV as the benchmark: does our forecast turn
   to drought before the Drought Monitor does for that planning region?

## 6. Running it in VS Code

```
git clone https://github.com/wchoo30/YPI-test
cd YPI-test/drought_rf
pip install -r requirements.txt
python -m pytest tests            # checks, ~1 min
python -m drought_rf              # synthetic demo, ~3 min
python fetch_data.py --sites 01188000,01193500
python -m drought_rf --data data/raw --split walk_forward
```
