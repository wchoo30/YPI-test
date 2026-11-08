"""Build the tables behind the 'Connecticut Drought Signals' charts.

Inputs:  usdm csv (repo data/ct_drought_monitor_2000_2026.csv), nClimGrid-Daily county CSV
Outputs: usdm_state, usdm_regions, rain_state, rain_counties, lead_lag CSVs in OUT.
Run from the drought_rf folder:  python scripts/build_dashboard_data.py USDM.csv COUNTY.csv OUTDIR
"""
import sys
import pandas as pd
sys.path.insert(0, '.')
from drought_rf.percentiles import doy_pools, variable_threshold_percentile

USDM, WX, OUT = sys.argv[1], sys.argv[2], sys.argv[3].rstrip('/') + '/'
u = pd.read_csv(USDM, dtype={'FIPS': str})
u['date'] = pd.to_datetime(u['ValidStart'])
u['region'] = u['County'].str.replace(' Planning Region', '').str.replace('Connecticut', 'CT')
st = u.groupby('date')[['D0', 'D1', 'D2', 'D3', 'D4']].mean().round(2).reset_index()
st.columns = ['date', 'd0', 'd1', 'd2', 'd3', 'd4']   # cumulative: d1 = % area in D1 or worse
st['date'] = st.date.dt.strftime('%Y-%m-%d'); st.to_csv(OUT + 'usdm_state.csv', index=False)
rg = u[['date', 'region', 'D1']].rename(columns={'D1': 'd1'}).copy()
rg['date'] = rg.date.dt.strftime('%Y-%m-%d')
rg.sort_values(['region', 'date']).to_csv(OUT + 'usdm_regions.csv', index=False)

w = pd.read_csv(WX, parse_dates=['date'])
w = w[w.prcp >= 0]
w['county'] = w.region_name.str.replace('CT: ', '').str.replace(' County', '')
rows = []
for c, g in w.groupby('county'):
    s = g.set_index('date')['prcp'].asfreq('D')
    idx = s.index
    ref = pd.Series((idx.year >= 1991) & (idx.year <= 2020), index=idx)
    pools = doy_pools(idx)
    df = pd.DataFrame(index=idx)
    for win in (30, 90, 365):
        r = s.rolling(win, min_periods=win).sum()
        df[f'p{win}'] = variable_threshold_percentile(r, ref_mask=ref, pools=pools)
        if win == 90:
            df['mm90'] = r
            doy = idx.dayofyear
            med = r[ref.values].groupby(doy[ref.values]).median()
            df['norm90'] = med.reindex(doy).values
    df['county'] = c
    rows.append(df.reset_index())
d = pd.concat(rows)
d = d[(d.date >= '2000-01-01') & (d.date.dt.dayofweek == 1)].dropna(subset=['p90'])  # Tuesdays = USDM week start
cty = d[['date', 'county', 'p90']].copy()
cty['p90'] = (cty.p90 * 100).round(1); cty['date'] = cty.date.dt.strftime('%Y-%m-%d')
cty.rename(columns={'p90': 'pct90'}).to_csv(OUT + 'rain_counties.csv', index=False)
sm = d.groupby('date')[['p30', 'p90', 'p365', 'mm90', 'norm90']].mean().reset_index()
for k in ['p30', 'p90', 'p365']:
    sm[k] = (sm[k] * 100).round(1)
sm['mm90'] = sm.mm90.round(0); sm['norm90'] = sm.norm90.round(0)
sm = sm.rename(columns={'p30': 'pct30', 'p90': 'pct90', 'p365': 'pct365', 'mm90': 'rain90_mm', 'norm90': 'normal90_mm'})
sm['date'] = sm.date.dt.strftime('%Y-%m-%d'); sm.to_csv(OUT + 'rain_state.csv', index=False)

U = st.assign(date=pd.to_datetime(st.date)).set_index('date')['d1']
S = sm.assign(date=pd.to_datetime(sm.date)).set_index('date')
ll = []
for lagw in range(0, 27):
    tgt = U.copy(); tgt.index = tgt.index - pd.Timedelta(weeks=lagw)
    j = S.join(tgt.rename('y'), how='inner').dropna()
    row = {'lead_weeks': lagw, 'n': len(j)}
    for k in ['pct30', 'pct90', 'pct365']:
        # sign flipped so "drier rain -> more drought area" reads as a positive number
        row[k.replace('pct', 'r') + 'd'] = round(-j[k].corr(j['y'], method='spearman'), 3)
    ll.append(row)
pd.DataFrame(ll).to_csv(OUT + 'lead_lag.csv', index=False)
print(pd.DataFrame(ll).head(8).to_string()); print(sm.tail(2).to_string())
