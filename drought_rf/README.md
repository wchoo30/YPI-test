# drought_rf: random forest drought predictions for Connecticut

A Python rebuild of the model in Heldmyer et al. (2026), *Predicting streamflow drought in
the conterminous United States using machine learning and a donor-gage approach,
1982–2020*, HESS 30:5925 (https://doi.org/10.5194/hess-30-5925-2026), extended from a
same-day nowcast to forecasts 7, 14 and 30 days ahead so it can feed ctwater.

Status (2026-11-08): two models. The gauge model (below) runs on synthetic data only so
far. The Drought Monitor level model (`drought_rf/usdm.py`) runs on real data; see
[RESULTS.md](RESULTS.md). 7 tests pass.

## What the paper does, in one paragraph

USGS converted daily streamflow at 3,198 gauges to day-of-year percentiles (Weibull
plotting position, 30-day window) and called a day "drought" when flow fell below the 20th
percentile. One random forest per gauge then classifies each day as drought or not from
130 weather predictors: soil moisture at 10–40 and 40–100 cm, precipitation, SPEI, PET,
min/max temperature and snow water equivalent, each as raw values and percentiles over
30-, 90- and 365-day windows, plus AMO/PDO/ENSO/PNA, sunspots and decimal date. Classes
are weighted 0.8/0.2, node size 10, no tuning. Training 1985–2015, testing 1982–84 and
2016–20, scored by Cohen's kappa. The most important predictors were deep and shallow soil
moisture percentiles and 90-day precipitation. For places with no gauge, a "donor gauge"
step borrows the models of the gauges whose importance patterns look most alike,
predicted from basin attributes. Mean kappa was about 0.40 for donor predictions.

Two things to keep in mind: the paper predicts *today's* drought from today's weather
(no lead time), and its drought is *streamflow* drought, not soil or crop drought.

## How predictions work here

```
            daily weather, soil moisture, snow (per gauge basin)      monthly AMO/PDO/ENSO/PNA
                              │                                                  │
            rolling 30/90/365-day sums or means                     lagged 1 month (publication delay)
                              │                                                  │
            day-of-year percentile + whole-record percentile                     │
                              └────────────── predictors at day t ───────────────┘
                                                   │   (+ optional: today's flow percentile)
USGS daily flow ─► day-of-year percentile ─► label at day t + lead: drought if < 20th pct
                                                   │
                       one model per gauge and lead (0, 7, 14, 30 days)
                                                   │
                      P(drought at t + lead) ─► threshold (0.5, or tuned for kappa)
                                                   │
             gauges: score vs observed flow       farm cells: donor-gauge average
```

* **Lead 0** reproduces the paper (a nowcast; useful for filling gaps and ungauged streams).
* **Leads 7–30** pair predictors known on day *t* with the label on day *t + lead*. Nothing
  after day *t* enters a row, so this is a real forecast. Drought is slow, so antecedent
  soil moisture and 90/365-day rainfall should carry most of the 30-day skill. Adding
  forecast rainfall (GEFS, CPC week 3–4) later is a column join.
* **Percentiles** are computed against the training years only, so the test period does
  not leak into its own thresholds.
* **Splits**: `--split paper` (train 1985–2015, test 1982–84 + 2016–20) to compare with the
  paper, or `--split walk_forward` (train on all years before each test year, 30-day
  embargo), which is ctwater's convention.
* **Farms** have no gauge, so they get predictions through the donor step
  (`drought_rf/donor.py`): predict which gauges' models fit a farm's catchment from its
  attributes, run those models on the farm's own weather, average.

## Models we can compare

| Model | Why it is here |
|---|---|
| `rf` random forest | The paper's model and settings (500 trees, node size 10, weights 0.8/0.2). Gini importance. |
| `lgbm` LightGBM | Same inputs, boosted trees; what ctwater already uses. Usually a little sharper. |
| `logistic` | Linear benchmark. If trees barely beat it, the signal is mostly in a few percentiles. |
| `climatology` | Drought frequency for that site and time of year. Brier skill is measured against this. |
| `persistence` | "Drought today means drought at t + lead." The baseline any forecast must beat at gauges. |

Worth adding once real data are in: a pooled model (all CT gauges in one forest with basin
attributes as predictors; more data per tree than 10–20 single-gauge models), quantile
regression forest for the flow percentile itself rather than a yes/no, and an LSTM if the
tree models plateau. SHAP values would answer the paper's own interpretability gap.

See [DATA.md](DATA.md) for every input's resolution, period and access.

## Running it

```
pip install -r requirements.txt
python -m drought_rf                                   # synthetic demo, ~3 min
python fetch_data.py --sites 01188000,01193500 ...     # on your machine; writes data/raw/
python -m drought_rf --data data/raw --split walk_forward --leads 0,7,14,30
python -m pytest tests
```
Outputs: `predictions.csv` (every test day, every model), `scores.csv` (kappa,
sensitivity, specificity, balanced accuracy, Brier, Brier skill per site/lead/model),
`rf_importance.csv`, `summary.csv`. Input layout is in `drought_rf/io.py`.

## Data I need from you

The cloud container this was built in cannot reach USGS, gridMET or NOAA, so either run
`fetch_data.py` locally and drop `data/raw/` into the project folder, or send the files.

1. **Which drought you want predicted.** Streamflow drought at gauges (the paper), or the
   ctwater farm water-balance target, or Drought Monitor / CT drought stage. The code is
   built for streamflow; the others swap in a different label table.
2. **The gauge list.** Unregulated Connecticut gauges with long daily records, ideally the
   GAGES-II reference set. The paper's own 3,198-gauge list (in its USGS data release)
   is the filter to use; see DATA.md. Examples: 01188000 Bunnell Brook, 01193500 Salmon River.
3. **Soil moisture**, the paper's top predictor, at two depths, daily since ~1980. Best
   options: ctwater's own soil bucket output per cell (if you have it), NOAA National Water
   Model retrospective, or NLDAS-2. Tell me which ctwater already has.
4. **Basin boundaries or the farm cell list.** Upstream basin polygons per gauge (the
   paper averaged weather over the basin; the fetch script uses the gauge point for now),
   and the 200 farm cell centroids for the donor step.
5. **Static attributes** per gauge and farm cell (GAGES-II basin characteristics, or soil,
   land cover, elevation from SSURGO / CDL / 3DEP that ctwater already computes).
6. Optional: snow water equivalent (UA 4 km SWE or Daymet), and access to the ctwater repo
   so this can live there and reuse its loaders.

## Differences from the paper

* Python/scikit-learn instead of R/caret; same RF settings. mtry and tree count were not
  reported, so sqrt(p) and 500 trees are used.
* P − PET percentiles stand in for SPEI. Sunspot number is left out. Day-of-year sine and
  cosine are added.
* Zero-flow handling is not implemented (CT reference gauges rarely go dry).
* Forecast leads, training-years-only percentiles, walk-forward splits and the
  persistence/climatology/LightGBM/logistic comparisons are additions.
