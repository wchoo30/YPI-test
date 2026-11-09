# How to use drought_rf

Everything below runs from a terminal inside VS Code (Terminal → New Terminal).
Commands assume you are in the `drought_rf` folder of the repo.

## 1. One-time setup

```bash
git clone https://github.com/wchoo30/YPI-test
cd YPI-test
git checkout drought-rf-prototype      # until the pull request is merged; after that, main branch has it
cd drought_rf
python -m venv .venv                   # optional but keeps packages tidy
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Check that it works (about 3 minutes)

```bash
python -m pytest tests                 # 6 checks, should say "6 passed"
python -m drought_rf                   # runs the whole model on FAKE data
```

The second command prints a score table. Those numbers are meaningless: it only proves
the pipeline runs. Output files land in `outputs/`.

## 3. Get real data

**Stream gauges + weather + climate indices** (needs normal internet, so run it on your computer):

```bash
python fetch_data.py --sites 01188000,01193500 --start 1980-01-01 --end 2025-12-31
```

This writes `data/raw/streamflow.csv`, `met.csv`, `teleconnections.csv` and `gauges.csv`.
Add more gauge numbers separated by commas (see DATA.md for how to pick them). It has
never been run before, so if it errors, send the error to Claude and it gets fixed.

**County rainfall from NOAA** (used for the charts; works anywhere):

```bash
python scripts/fetch_nclimgrid_counties.py data/raw/ct_county_daily.csv
```

Downloaded data is ignored by git on purpose: each person re-downloads it.

## 4. Run the real model

```bash
python -m drought_rf --data data/raw --split walk_forward --leads 0,7,14,30
```

Options:

| Option | What it does |
|---|---|
| `--leads 0,7,14,30` | How many days ahead to predict. 0 = is there drought today (the paper's setup). |
| `--split walk_forward` | Train on all earlier years, test one year at a time (how a live model works). `--split paper` reproduces the paper's 1985–2015 / 2016–2020 split. |
| `--models rf,lgbm,logistic,climatology,persistence` | Which models to run. `rf` is the random forest; the rest are comparisons. |
| `--use-flow` | Also use today's streamflow as a predictor (stronger forecasts at gauges). |
| `--threshold 0.5` | Probability above which a day is called "drought". |
| `--out outputs` | Where results go. |

## 4b. Predict Drought Monitor levels per planning region (runs on real data now)

```bash
python scripts/fetch_nclimgrid_counties.py data/raw/ct_county_daily.csv
python -m drought_rf.usdm --usdm ../data/ct_drought_monitor_2000_2026.csv --weather data/raw/ct_county_daily.csv
```

Takes about 7 minutes. Predicts None / D0 / D1 / D2+ for each of the 9 planning regions
0, 1, 2 and 4 weeks ahead (`--leads 0,1,2,4`), testing each year 2010–2025 on a model
trained only on earlier years. Results go to `outputs/usdm/`: `summary.csv` (scores),
`predictions.csv` (every region-week with each level's probability), `rf_importance.csv`.
The latest results and what they mean are in `RESULTS.md`.

## 5. Read the results (in `outputs/`)

| File | What's in it |
|---|---|
| `summary.csv` | Average score per model and lead. Start here. |
| `scores.csv` | The same per gauge. |
| `predictions.csv` | Every test day: actual drought (`y`) and each model's probability. |
| `rf_importance.csv` | Which predictors the random forest relied on. |

How to read the scores: **kappa** 0 = no better than chance, 1 = perfect (the paper got
~0.4–0.6). **sensitivity** = share of drought days caught. **specificity** = share of
normal days correctly left alone. **brier_skill** above 0 = better than "usual drought
rate for this time of year". The forest is only useful if it beats `persistence` at
7–30 days.

Quick look in Python or a notebook:

```python
import pandas as pd
s = pd.read_csv("outputs/summary.csv")
s.pivot(index="lead", columns="model", values="kappa")
```

## 5b. See the results as charts

Open `results.ipynb` in VS Code (install the **Jupyter** extension once), pick your Python
as the kernel, and click **Run All**. It reads `outputs/usdm/` and `outputs/` and draws:
accuracy by model, new-drought warnings vs false alarms, predicted vs actual over time for
one region, a predicted-vs-actual table and the top predictors. Change `REGION` and
`LEAD` in the second cell to look at another region or horizon. The copy in the repo is
saved with the charts from the 2026-11-08 run, so you can see them before running anything.

## 6. Rebuild the chart data

```bash
python scripts/build_dashboard_data.py ../data/ct_drought_monitor_2000_2026.csv data/raw/ct_county_daily.csv outputs/
```

## 7. Working as a group

- Make a branch for each change: `git checkout -b my-change`, commit, `git push -u origin my-change`, then open a pull request on GitHub.
- Run `python -m pytest tests` before pushing.
- Never commit data files or passwords: the repo is public.

## Where things are explained

- `RESULTS.md`: latest real-data results
- `results.ipynb`: charts of the results

- `README.md`: what the model is and the prediction diagram
- `WALKTHROUGH.md`: every file and each step of a prediction
- `DATA.md`: each dataset's resolution, years and access
- `INDICATORS.md`: the CT drought indicator table
