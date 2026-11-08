# Results so far

## Drought Monitor level per planning region (real data, run 2026-11-08)

Predicts each Connecticut planning region's Drought Monitor level (None, D0, D1, D2 or
worse) 0, 1, 2 and 4 weeks ahead. Data: the repo's Drought Monitor export (2000–2026)
and NOAA nClimGrid county weather (to Feb 21, 2026). Walk-forward test: each year
2010–2025 is predicted by a forest trained only on earlier years (4-week gap).
7,515 test region-weeks per lead.

```
python scripts/fetch_nclimgrid_counties.py data/raw/ct_county_daily.csv
python -m drought_rf.usdm --usdm ../data/ct_drought_monitor_2000_2026.csv --weather data/raw/ct_county_daily.csv
```

| Weeks ahead | Model | Exact level | Within one level | New droughts caught | False alarm rate |
|---|---|---|---|---|---|
| 1 | Same as this week (persistence) | 92 % | 100 % | 0 % | 0 % |
| 1 | Forest, weather + current level | 86 % | 100 % | **56 %** | 2.8 % |
| 1 | Forest, weather only | 68 % | 92 % | 65 % | 14.5 % |
| 2 | Persistence | 85 % | 99 % | 0 % | 0 % |
| 2 | Forest, weather + current level | 77 % | 98 % | **54 %** | 5.2 % |
| 4 | Persistence | 75 % | 96 % | 0 % | 0 % |
| 4 | Forest, weather + current level | 67 % | 94 % | **31 %** | 7.5 % |
| 4 | Forest, weather only | 64 % | 89 % | 43 % | 14.0 % |

"New droughts caught": weeks where a region is not in D1 now but is at D1 or worse at
the target week (99 cases at 1 week, 181 at 2, 314 at 4). The forest raises an alarm
when its probability of D1 or worse is at least 30 %. "False alarm rate": share of
no-drought-coming weeks with an alarm.

What this says:
- **On exact level, the forest does not beat "same as this week" yet.** Drought Monitor
  levels change slowly, so copying this week is right most of the time.
- **It does warn of new droughts, which copying this week never can.** At 2 weeks ahead it
  flags about half of new D1 droughts with alarms on about 1 in 20 quiet weeks.
- Most important predictors: the current level and D1 area, then 90-day rainfall minus
  evaporation, 90-day temperature and 90-day rainfall percentiles (`rf_importance.csv`).

Caveats: county weather stands in for each planning region (the two don't line up
exactly); rainfall percentiles use 1991–2020 as the reference, which overlaps some test
years (no labels are involved); D3 is merged into D2+ because it covered half a region
in only 67 region-weeks and D4 never did.

Next improvements to try: predict the *change* in level instead of the level, tune the
alarm threshold per lead, add PRISM rainfall, groundwater and streamflow.

## Gauge model (streamflow drought)

Not run on real data yet: needs `fetch_data.py` (USGS streamflow) run on a computer with
normal internet.
