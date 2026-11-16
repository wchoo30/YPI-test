# How good is the model? Reading the predictions and the scores

## 1. What one prediction says

The Drought Monitor model makes one forecast per **planning region**, per **issue week**, per **lead**.
A row in `outputs/usdm/predictions.csv` (backtest) or `forecasts/log.csv` (live) reads like this:

| column | example | meaning |
|---|---|---|
| region / site | Northwest Hills | one of CT's 9 planning regions |
| issue date (`date`) | 2024-09-03 | the Drought Monitor map the forecast starts from. Weather up to this day is used, nothing after |
| lead | 2 weeks | how far ahead |
| target date | 2024-09-17 | the map being forecast |
| level now | D0 | the region's level on the issue map (worst level covering at least half the region) |
| P(None), P(D0), P(D1), P(D2+) | 0.05, 0.40, 0.45, 0.10 | the forest's probabilities. They add up to 1 |
| forecast level | D1 | the most likely level (largest probability) |
| P(D1+) | 0.55 | P(D1) + P(D2+): the chance the region is in **moderate drought or worse** on the target map |
| new drought warning | yes | the region is below D1 now and P(D1+) is at least 0.3 |

So the example says: "Northwest Hills is D0 today. Two weeks from now there is a 55 % chance it is at
D1 or worse; D1 is the single most likely level." The 0.3 warning level is a choice; section 4 shows
how to pick it.

The **gauge model** is simpler: for each stream gauge and day, one probability that flow will be below
its 20th percentile for that time of year on the target day (`rf`, `climatology` and `persistence`
columns in `outputs/predictions.csv`). The yes/no call is P ≥ `--threshold` (default 0.5).

## 2. How the scores are made honest

* **Walk-forward testing.** For each test year the model is trained only on earlier years, with a gap
  before the test year, then scored on that year. Every score is for a forecast the model could have
  made at the time.
* **Baselines.** Each score is shown for *persistence* (the level stays the same) and *climatology*
  (the usual level/chance for that region and month). A forecast is only useful where it beats them.
* **Uncertainty.** Drought weeks come in runs, so 800 drought weeks are not 800 independent tests.
  The scorecard resamples **whole years** 300 times and reports the middle 90 % of each score
  (`_lo`, `_hi` columns, brackets in the notebook). A wide range means the score rests on a few
  drought years.
* **Live log.** `python -m drought_rf.live` writes forecasts *before* the answer exists and never
  rewrites them, so `forecasts/verification_summary.csv` is a true out-of-sample test.

## 3. What each score means

Yes/no scores for "D1 or worse" (gauge model: "below 20th percentile"):

| score | question it answers | good |
|---|---|---|
| hit rate | of the droughts that happened, what share were called? | high |
| false alarm ratio | of the drought calls made, what share were wrong? | low |
| false alarm rate | of the quiet weeks, what share were wrongly called? | low |
| CSI (critical success index) | hits / (hits + misses + false alarms); ignores the easy quiet weeks | high |
| Peirce skill | hit rate minus false alarm rate; 0 = no skill | high |
| kappa | agreement beyond chance; 0 = chance, 1 = perfect (the paper's main score) | high |

Probability scores (use the probability itself, no threshold):

| score | question | good |
|---|---|---|
| Brier score | average squared gap between the probability and what happened (0 or 1) | low |
| Brier skill | 1 − Brier / Brier of climatology. 0 = no better than "usual chance for this month"; below 0 = worse | above 0 |
| ROC AUC | if you pick one drought week and one quiet week, how often does the drought week get the higher probability? 0.5 = coin flip | near 1 |
| reliability | when the model says 30 %, does drought happen about 30 % of the time? | dots on the diagonal |

Level scores (Drought Monitor model only):

| score | meaning |
|---|---|
| exact level | share of forecasts with exactly the right level |
| within one level | share no more than one level off |
| mean level error | average number of levels off |
| bias | average forecast level minus actual; above 0 = forecasts too dry |
| weighted kappa | like kappa, but being two levels off counts more than one |

New-drought scores (Drought Monitor model): an **onset** is a region below D1 now that is at D1 or
worse on the target map. Persistence can never catch one, so this is where the forest has to earn its
keep. *Onset caught* = share of onsets with a warning; *onset false alarm rate* = share of quiet weeks
that got a warning.

## 4. Results on real data

Drought Monitor model, walk-forward test 2010-2025 (7,515 region-weeks per lead), forest with
current level ("rf_weather_plus_now"). Brackets: 90 % range across resampled years.

| | 1 week | 2 weeks | 4 weeks |
|---|---|---|---|
| exact level, forest | 86 % [83-89] | 77 % [71-82] | 67 % [60-74] |
| exact level, persistence | 92 % | 85 % | 75 % |
| mean level error, forest | 0.15 | 0.25 | 0.40 levels |
| D1+ hit rate (alarm at 0.3) | 96 % | 92 % | 77 % |
| D1+ false alarm ratio | 19 % | 30 % | 43 % |
| D1+ Brier skill, forest | 0.74 [0.68-0.79] | 0.60 [0.52-0.66] | 0.42 [0.33-0.48] |
| D1+ Brier skill, persistence | 0.82 | 0.66 | 0.41 |
| ROC AUC, forest / persistence | 0.99 / 0.95 | 0.97 / 0.91 | 0.92 / 0.85 |
| new droughts caught | 56 % [39-73] | 54 % [37-70] | 31 % [19-42] |
| quiet weeks with a false warning | 3 % | 5 % | 8 % |

What this says:

1. **Levels: persistence still wins.** Drought Monitor levels change slowly, so "same as this week"
   gets the exact level more often at every lead. The forest is rarely far off (within one level
   94-100 % of the time, bias near zero), but it changes level too eagerly.
2. **Ranking risk: the forest wins.** Its ROC AUC is higher than persistence at every lead: it orders
   regions from least to most likely to be in D1 better.
3. **Probabilities: about a tie.** Brier skill is a bit below persistence at 1-2 weeks and level at 4
   weeks. The reliability table shows why: below 30 % the forest is too alarmed (it says 15 %, D1
   happens 3 % of the time) and above 40 % not alarmed enough (says 45 %, happens 66 %). That comes
   from the class weighting used to train it. Re-calibrating on earlier years closes part of the gap
   (0.63 vs 0.68 at 2 weeks) but not all of it.
4. **New droughts: the forest's real value.** It warns of about half of new D1 droughts 1-2 weeks
   ahead with few false warnings, which persistence cannot do at all. The ranges are wide (37-70 % at
   2 weeks) because there are only ~10 drought episodes in the test years.
5. **It depends on the year.** 2015-2017, 2020, 2022, 2024 are caught well (hit rate 89-100 %). 2012
   onsets and the brief 2023 D1 were missed entirely, and 2023 has negative Brier skill.
6. **Threshold trade-off (2 weeks):** alarm at 0.2 catches 97 % of D1 weeks with 44 % of alarms
   wrong; 0.3 gives 92 % / 30 %; 0.4 gives 82 % / 21 % (the best kappa, 0.77).

The gauge model's scorecard is run on your machine (its data isn't in the cloud copy):
`python -m drought_rf.scorecard --gauge outputs/predictions.csv`. From your 2-gauge run, the
30-day forest had Brier skill +0.05 but caught only 11 % of drought days at the 0.5 cut, so look at
its threshold table and AUC before reading much into kappa.

## 5. Running it

```
python -m drought_rf.usdm --usdm ../data/ct_drought_monitor_2000_2026.csv --weather data/raw/ct_county_daily.csv --out outputs/usdm
python -m drought_rf.scorecard --usdm outputs/usdm/predictions.csv
python -m drought_rf.scorecard --gauge outputs/predictions.csv      # after the gauge model
```

Tables land in `outputs/scorecard/`: `*_scorecard.csv` (all scores with ranges), `*_reliability.csv`,
`*_thresholds.csv`, `*_by_year.csv`, `*_by_region.csv` / `gauge_by_site.csv`. Part 3 of `results.ipynb`
draws them.
