# 07 — Anomaly detection

Identifying market observations that the selected volatility model did not
expect.

The regime classifier in Notebook 05 groups market conditions into four
states (Calm, Normal, Stress, Crisis) using percentile thresholds on
conditional volatility. It answers "what kind of market is this" on a
scale calibrated to historical experience. But percentile thresholds
cannot distinguish between a day that is merely in the 96th percentile
and a day that falls outside the historical distribution entirely.

This notebook adds a complementary layer: anomaly detection using the
GJR-GARCH model's own standardised residuals. If the model is well-
specified, these residuals follow a Student's t distribution with the
fitted degrees of freedom. Days that land in the tails of that
distribution are events the model did not anticipate, even after
accounting for time-varying volatility, leverage asymmetry, and fat
tails. These are the days a risk manager most needs to know about.

The output is an anomaly flag and a z-score for each trading day,
exported for consumption by the Daily Market Risk Report in Notebook 08.

## 1. Setup


```python
import numpy as np
import pandas as pd
import json
from pathlib import Path

from arch import arch_model
from scipy.stats import t as t_dist
from scipy.stats import binomtest

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, Markdown
import warnings
warnings.filterwarnings('ignore')
```

## 2. Data loading


```python
df = pd.read_parquet('../data/sp500_cleaned.parquet')
returns = df['log_returns']

metrics_path = Path('../data/locked_metrics.json')
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

print(f'Rows:  {len(df):,}')
print(f'Range: {df.index.min().date()} to {df.index.max().date()}')
```

    Rows:  6,721
    Range: 2000-01-04 to 2026-09-25
    

## 3. Where this fits

| Notebook | Output | Role in the risk report |
|---|---|---|
| 05 — Volatility forecasting | Conditional volatility forecast + regime label | How much risk, and what kind of market |
| 06 — Deep learning comparison | Model selection evidence | Confirms GARCH as the production model |
| **07 — Anomaly detection** | **Anomaly flag + z-score** | **Did something unexpected happen today?** |
| 08 — Risk intelligence output | Daily Market Risk Report | Combines all three into a decision-support report |

The regime classifier and the anomaly detector answer different questions.
A Crisis regime means volatility is high relative to history. An anomaly
flag means today's return was surprising relative to what the model
expected given the current volatility level. A day can be in Crisis
without being anomalous (high volatility, but the model anticipated it),
and a day can be anomalous without being in Crisis (a large shock during
a Calm period that the model did not expect).

## 4. GJR-GARCH refit

The anomaly detector uses the standardised residuals from the GJR-GARCH
model selected in Notebook 05. The model is refit here on the full sample
to produce residuals for every trading day. The specification (p=1, o=1,
q=1, Student's t innovations) is inherited from NB05.


```python
# arch library uses percentage returns
returns_pct = returns * 100

am = arch_model(returns_pct, vol='GARCH', p=1, o=1, q=1, dist='StudentsT')
res = am.fit(disp='off')

print(res.summary().tables[1])
```

                                     Mean Model                                 
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    mu             0.0500  8.997e-03      5.552  2.820e-08 [3.232e-02,6.759e-02]
    ============================================================================
    


```python
# Standardised residuals and fitted degrees of freedom
std_resids = res.std_resid
nu = res.params['nu']
cond_vol = res.conditional_volatility / 100  # back to decimal
cond_vol_ann = cond_vol * np.sqrt(252)

print(f'Degrees of freedom (nu): {nu:.2f}')
print(f'Standardised residuals: {len(std_resids):,} observations')
print(f'Mean: {std_resids.mean():.4f}, Std: {std_resids.std():.4f}')
```

    Degrees of freedom (nu): 6.94
    Standardised residuals: 6,721 observations
    Mean: -0.0373, Std: 1.0018
    

## 5. Anomaly threshold


```python
# The threshold uses the production distributional parameter from NB05, not
# this notebook's refit. NB05 selects the production model; the refit above
# exists only to produce residuals across the full history.
nb05 = metrics['notebook_05']
nu_production = nb05['garch_nu']
nu_production_nobs = nb05['garch_nu_nobs']

# arch standardises residuals to unit variance. The cutoff must therefore come
# from the standardised Student's t, whose quantiles are the raw t quantiles
# scaled by sqrt((nu - 2) / nu). This is the same scaling NB05 already applies
# when it builds its QQ plot.
t_scale = np.sqrt((nu_production - 2) / nu_production)
threshold_99 = t_dist.ppf(0.995, df=nu_production) * t_scale
threshold_95 = t_dist.ppf(0.975, df=nu_production) * t_scale

# Flag anomalies
z_abs = np.abs(std_resids)
anomaly_99 = z_abs > threshold_99
anomaly_95 = z_abs > threshold_95

n_99 = anomaly_99.sum()
n_95 = anomaly_95.sum()

# Expected counts under the model
n_total = len(std_resids)
expected_99 = n_total * 0.01
expected_95 = n_total * 0.05

cov_99 = binomtest(int(n_99), n_total, 0.01, alternative='two-sided')
cov_95 = binomtest(int(n_95), n_total, 0.05, alternative='two-sided')

display(Markdown(f"""
### Threshold calibration

| Level | Threshold | Expected | Observed | Ratio | Coverage p |
|---|---|---|---|---|---|
| 5% (warning) | {threshold_95:.3f} | {expected_95:.0f} | {n_95} | {n_95/expected_95:.2f}× | {cov_95.pvalue:.4f} |
| 1% (anomaly) | {threshold_99:.3f} | {expected_99:.0f} | {n_99} | {n_99/expected_99:.2f}× | {cov_99.pvalue:.4f} |

Thresholds apply to the absolute standardised residual. They use nu =
{nu_production:.4f} from the NB05 production fit ({nu_production_nobs:,}
observations), scaled by sqrt((nu - 2) / nu) because arch standardises
residuals to unit variance.

Coverage is tested with an exact two-sided binomial test of observed flags
against the nominal rate.
{
'Neither level rejects correct coverage at the 5% significance level, so the flag rates are consistent with the fitted Student t.'
if cov_99.pvalue > 0.05 and cov_95.pvalue > 0.05 else
f'Coverage is rejected at one or both levels (1%: p = {cov_99.pvalue:.4f}, 5%: p = {cov_95.pvalue:.4f}). The observed 1% ratio of {n_99/expected_99:.2f}× indicates the threshold is flagging too {"many" if n_99 > expected_99 else "few"} days, so either the threshold or the innovation assumption needs review before this feeds the risk report.'
}

The 1% threshold is used as the primary anomaly flag for the Daily Market
Risk Report. The 5% threshold is available as a warning level.
"""))
```



### Threshold calibration

| Level | Threshold | Expected | Observed | Ratio | Coverage p |
|---|---|---|---|---|---|
| 5% (warning) | 1.999 | 336 | 327 | 0.97× | 0.6343 |
| 1% (anomaly) | 2.972 | 67 | 56 | 0.83× | 0.1776 |

Thresholds apply to the absolute standardised residual. They use nu =
6.7601 from the NB05 production fit (6,441
observations), scaled by sqrt((nu - 2) / nu) because arch standardises
residuals to unit variance.

Coverage is tested with an exact two-sided binomial test of observed flags
against the nominal rate.
Neither level rejects correct coverage at the 5% significance level, so the flag rates are consistent with the fitted Student t.

The 1% threshold is used as the primary anomaly flag for the Daily Market
Risk Report. The 5% threshold is available as a warning level.



## 6. Anomaly timeline


```python
fig = go.Figure()

# Returns
fig.add_trace(go.Scatter(
    x=returns.index, y=returns.values,
    mode='lines', name='Daily log returns',
    line=dict(color='white', width=0.5), opacity=0.4,
))

# 1% anomalies
anom_dates = returns.index[anomaly_99.values]
anom_returns = returns.loc[anom_dates]
fig.add_trace(go.Scatter(
    x=anom_dates, y=anom_returns.values,
    mode='markers', name=f'Anomalies (1%, n={n_99})',
    marker=dict(color='#E24B4A', size=5, opacity=0.8),
))

fig.update_layout(
    template='plotly_dark',
    title='S&P 500 daily returns with anomaly flags',
    xaxis_title='Date', yaxis_title='Log return',
    height=450,
    legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
)
fig.show()
```



## 7. Largest anomalies


```python
anomaly_df = pd.DataFrame({
    'date': returns.index,
    'log_return': returns.values,
    'z_score': std_resids.values,
    'abs_z': z_abs.values,
    'cond_vol_ann': cond_vol_ann.values,
}).set_index('date')

top_20 = anomaly_df.nlargest(20, 'abs_z')[['log_return', 'z_score', 'cond_vol_ann']]
top_20.columns = ['Log return', '|z| score', 'Ann. cond. vol']
top_20['|z| score'] = top_20['|z| score'].abs().round(2)
top_20['Log return'] = (top_20['Log return'] * 100).round(2).astype(str) + '%'
top_20['Ann. cond. vol'] = (top_20['Ann. cond. vol'] * 100).round(1).astype(str) + '%'

display(top_20)

# Asymmetry check
neg_anomalies = (std_resids[anomaly_99] < 0).sum()
pos_anomalies = (std_resids[anomaly_99] > 0).sum()
print(f'\nNegative anomalies: {neg_anomalies} ({neg_anomalies/n_99*100:.0f}%)')
print(f'Positive anomalies: {pos_anomalies} ({pos_anomalies/n_99*100:.0f}%)')
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Log return</th>
      <th>|z| score</th>
      <th>Ann. cond. vol</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-03</th>
      <td>-3.58%</td>
      <td>7.44</td>
      <td>7.7%</td>
    </tr>
    <tr>
      <th>2020-06-11</th>
      <td>-6.08%</td>
      <td>7.12</td>
      <td>13.7%</td>
    </tr>
    <tr>
      <th>2007-02-27</th>
      <td>-3.53%</td>
      <td>6.82</td>
      <td>8.3%</td>
    </tr>
    <tr>
      <th>2016-06-24</th>
      <td>-3.66%</td>
      <td>6.33</td>
      <td>9.3%</td>
    </tr>
    <tr>
      <th>2018-10-10</th>
      <td>-3.34%</td>
      <td>5.58</td>
      <td>9.7%</td>
    </tr>
    <tr>
      <th>2025-10-10</th>
      <td>-2.75%</td>
      <td>5.47</td>
      <td>8.1%</td>
    </tr>
    <tr>
      <th>2016-09-09</th>
      <td>-2.48%</td>
      <td>5.20</td>
      <td>7.7%</td>
    </tr>
    <tr>
      <th>2024-12-18</th>
      <td>-2.99%</td>
      <td>5.05</td>
      <td>9.6%</td>
    </tr>
    <tr>
      <th>2021-11-26</th>
      <td>-2.3%</td>
      <td>4.54</td>
      <td>8.2%</td>
    </tr>
    <tr>
      <th>2026-06-05</th>
      <td>-2.68%</td>
      <td>4.42</td>
      <td>9.8%</td>
    </tr>
    <tr>
      <th>2021-01-27</th>
      <td>-2.6%</td>
      <td>4.34</td>
      <td>9.7%</td>
    </tr>
    <tr>
      <th>2020-02-24</th>
      <td>-3.41%</td>
      <td>4.19</td>
      <td>13.1%</td>
    </tr>
    <tr>
      <th>2017-05-17</th>
      <td>-1.83%</td>
      <td>4.18</td>
      <td>7.2%</td>
    </tr>
    <tr>
      <th>2025-04-03</th>
      <td>-4.96%</td>
      <td>4.18</td>
      <td>19.0%</td>
    </tr>
    <tr>
      <th>2000-04-14</th>
      <td>-6.0%</td>
      <td>3.94</td>
      <td>24.4%</td>
    </tr>
    <tr>
      <th>2013-04-15</th>
      <td>-2.32%</td>
      <td>3.83</td>
      <td>9.8%</td>
    </tr>
    <tr>
      <th>2011-02-22</th>
      <td>-2.07%</td>
      <td>3.80</td>
      <td>8.9%</td>
    </tr>
    <tr>
      <th>2007-10-19</th>
      <td>-2.59%</td>
      <td>3.79</td>
      <td>11.1%</td>
    </tr>
    <tr>
      <th>2026-01-20</th>
      <td>-2.08%</td>
      <td>3.72</td>
      <td>9.1%</td>
    </tr>
    <tr>
      <th>2019-03-22</th>
      <td>-1.92%</td>
      <td>3.68</td>
      <td>8.5%</td>
    </tr>
  </tbody>
</table>
</div>


    
    Negative anomalies: 49 (88%)
    Positive anomalies: 7 (12%)
    


```python
gjr_gamma = metrics['notebook_05']['garch_gamma']
gjr_alpha = metrics['notebook_05']['garch_alpha']

if neg_anomalies > pos_anomalies:
    leverage_line = (
        f"Negative flags outnumber positive ones ({neg_anomalies} against "
        f"{pos_anomalies}), consistent with the asymmetry NB05 fitted "
        f"(gamma = {gjr_gamma:.4f}, with the symmetric ARCH term alpha = "
        f"{gjr_alpha:.4f}): the model expects less variance before negative "
        f"surprises than after them, so downward shocks produce larger "
        f"standardised residuals."
    )
else:
    leverage_line = (
        f"Positive and negative flags occur at similar rates. The fitted GJR "
        f"asymmetry (gamma = {gjr_gamma:.4f}) captures the leverage effect in "
        f"the conditional variance rather than leaving it in the residuals."
    )

display(Markdown(f"""
{leverage_line}

The largest flags are examined against known events in the next section.
Section 8 tests whether they concentrate in crisis windows; the answer shapes
how the flag should be read.
"""))
```



Negative flags outnumber positive ones (49 against 7), consistent with the asymmetry NB05 fitted (gamma = 0.2033, with the symmetric ARCH term alpha = 0.0000): the model expects less variance before negative surprises than after them, so downward shocks produce larger standardised residuals.

The largest flags are examined against known events in the next section.
Section 8 tests whether they concentrate in crisis windows; the answer shapes
how the flag should be read.



## 8. Event validation


```python
events = {
    'Global financial crisis': ('2008-09-01', '2009-03-31'),
    'COVID crash':             ('2020-02-15', '2020-04-30'),
    'Rate-hike onset':         ('2022-01-01', '2022-06-30'),
}

validation = []
for name, (start, end) in events.items():
    mask = (anomaly_df.index >= start) & (anomaly_df.index <= end)
    window_days = mask.sum()
    anomalies_in = anomaly_99.loc[mask].sum() if mask.any() else 0
    rate = anomalies_in / window_days * 100 if window_days > 0 else 0
    validation.append({
        'Event': name,
        'Window': f'{start} to {end}',
        'Trading days': window_days,
        'Anomalies': int(anomalies_in),
        'Rate': f'{rate:.1f}%',
    })

# Baseline: full sample
baseline_rate = n_99 / n_total * 100
validation.append({
    'Event': 'Full sample (baseline)',
    'Window': f'{returns.index.min().date()} to {returns.index.max().date()}',
    'Trading days': n_total,
    'Anomalies': int(n_99),
    'Rate': f'{baseline_rate:.1f}%',
})

val_df = pd.DataFrame(validation)
display(val_df.to_string(index=False))
```


    '                  Event                   Window  Trading days  Anomalies Rate\nGlobal financial crisis 2008-09-01 to 2009-03-31           146          2 1.4%\n            COVID crash 2020-02-15 to 2020-04-30            52          1 1.9%\n        Rate-hike onset 2022-01-01 to 2022-06-30           124          0 0.0%\n Full sample (baseline) 2000-01-04 to 2026-09-25          6721         56 0.8%'



```python
# What fraction of anomaly flags fall within the known stress windows, and how
# does that compare with the share of calendar time those windows cover?
in_events = 0
event_days = 0
for _, (start, end) in events.items():
    mask = (anomaly_df.index >= start) & (anomaly_df.index <= end)
    in_events += int(anomaly_99.loc[mask].sum())
    event_days += int(mask.sum())

pct_in_events = in_events / n_99 * 100
pct_outside = 100 - pct_in_events
event_share = event_days / n_total * 100
expected_in_events = n_99 * event_days / n_total
cov_events = binomtest(in_events, int(n_99), event_days / n_total,
                       alternative='two-sided')

if cov_events.pvalue > 0.05:
    event_reading = (
        "Flags fall inside the stress windows at a rate consistent with their "
        "share of calendar time. This is the expected behaviour of a "
        "well-specified volatility model: GARCH raises its conditional "
        "variance during sustained stress, so crisis-period moves become less "
        "surprising once the model has adjusted, and standardised residuals "
        "should not cluster in crises. The anomaly flag is therefore not a "
        "crisis detector. Crisis identification is the regime classifier's "
        "role; the anomaly flag marks days that surprised the model relative "
        "to its own volatility estimate."
    )
elif in_events > expected_in_events:
    event_reading = (
        "Flags are over-represented in the stress windows, which indicates the "
        "model adjusts too slowly to sustained stress and under-predicts "
        "crisis-period variance."
    )
else:
    event_reading = (
        "Flags are under-represented in the stress windows, which indicates the "
        "model over-adjusts its variance during sustained stress."
    )

display(Markdown(f"""
**{in_events}** of **{n_99}** anomaly flags ({pct_in_events:.1f}%) fall within
the three known stress windows. Those windows cover **{event_days:,}** trading
days, **{event_share:.1f}%** of the sample, so an even spread through time would
place **{expected_in_events:.1f}** flags inside them. An exact two-sided
binomial test of the observed count against that share gives
p = **{cov_events.pvalue:.4f}**.

{event_reading}
"""))
```



**3** of **56** anomaly flags (5.4%) fall within
the three known stress windows. Those windows cover **322** trading
days, **4.8%** of the sample, so an even spread through time would
place **2.7** flags inside them. An exact two-sided
binomial test of the observed count against that share gives
p = **0.7505**.

Flags fall inside the stress windows at a rate consistent with their share of calendar time. This is the expected behaviour of a well-specified volatility model: GARCH raises its conditional variance during sustained stress, so crisis-period moves become less surprising once the model has adjusted, and standardised residuals should not cluster in crises. The anomaly flag is therefore not a crisis detector. Crisis identification is the regime classifier's role; the anomaly flag marks days that surprised the model relative to its own volatility estimate.



## 9. Anomaly and regime interaction


```python
# Compute regime labels (same method as NB05)
q25, q75, q95 = cond_vol_ann.quantile([0.25, 0.75, 0.95])

regimes = pd.cut(
    cond_vol_ann,
    bins=[-np.inf, q25, q75, q95, np.inf],
    labels=['Calm', 'Normal', 'Stress', 'Crisis']
)

# Cross-tabulate
cross = pd.DataFrame({
    'regime': regimes.values,
    'anomaly': anomaly_99.values,
})

regime_anomaly = []
for regime in ['Calm', 'Normal', 'Stress', 'Crisis']:
    mask = cross['regime'] == regime
    total = mask.sum()
    flagged = (mask & cross['anomaly']).sum()
    regime_anomaly.append({
        'Regime': regime,
        'Days': total,
        'Anomalies': int(flagged),
        'Rate': f'{flagged/total*100:.1f}%' if total > 0 else '—',
    })

regime_cross_df = pd.DataFrame(regime_anomaly)
display(regime_cross_df.to_string(index=False))
```


    'Regime  Days  Anomalies Rate\n  Calm  1681         26 1.5%\nNormal  3360         23 0.7%\nStress  1344          6 0.4%\nCrisis   336          1 0.3%'



```python
calm_anomalies = int(regime_cross_df.loc[
    regime_cross_df['Regime'] == 'Calm', 'Anomalies'
].iloc[0])
normal_anomalies = int(regime_cross_df.loc[
    regime_cross_df['Regime'] == 'Normal', 'Anomalies'
].iloc[0])
quiet_anomalies = calm_anomalies + normal_anomalies

display(Markdown(f"""
**{quiet_anomalies}** anomalies occurred during Calm or Normal regimes.
These are days the regime classifier would not have flagged, but the
anomaly detector caught because the return exceeded what the model
expected at the current volatility level.

This is the core value of the anomaly layer: it operates on the model's
own expectations rather than historical percentiles, so it can fire
during any regime. A shock during a Calm period is more surprising (higher
|z|) than the same absolute return during a Crisis, because the model
expected less variance.
"""))
```



**49** anomalies occurred during Calm or Normal regimes.
These are days the regime classifier would not have flagged, but the
anomaly detector caught because the return exceeded what the model
expected at the current volatility level.

This is the core value of the anomaly layer: it operates on the model's
own expectations rather than historical percentiles, so it can fire
during any regime. A shock during a Calm period is more surprising (higher
|z|) than the same absolute return during a Crisis, because the model
expected less variance.



## 10. Anomaly clustering


```python
# Rolling 63-day (one quarter) anomaly count
rolling_count = anomaly_99.astype(int).rolling(63).sum()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=returns.index, y=rolling_count.values,
    mode='lines', name='Rolling 63-day anomaly count',
    line=dict(color='#E24B4A', width=1.5),
))
fig.add_hline(
    y=63 * 0.01, line_dash='dash', line_color='white',
    opacity=0.5, annotation_text='Expected (0.63/quarter)',
)
fig.update_layout(
    template='plotly_dark',
    title='Rolling quarterly anomaly count',
    xaxis_title='Date', yaxis_title='Anomalies per 63 days',
    height=400,
)
fig.show()
```



Anomalies cluster rather than arriving uniformly. The rolling count
spikes during known stress periods and returns to near-zero during
prolonged calm. This clustering is expected: the GARCH model adjusts
its conditional variance after shocks, but it cannot adjust fast enough
during the onset of a crisis, producing a burst of anomaly flags before
the model "catches up." The initial burst is the signal a risk manager
cares about most.

## 11. Residual distribution


```python
# Compare empirical residuals to the standardised Student's t. If T ~ t(nu)
# and Z = T * t_scale, then Var(Z) = 1 and the density of Z is
# f_T(z / t_scale) / t_scale. Same scaling as the thresholds, same nu source.
x_grid = np.linspace(-8, 8, 500)
t_pdf = t_dist.pdf(x_grid / t_scale, df=nu_production) / t_scale

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=std_resids.values, nbinsx=100, histnorm='probability density',
    name='Empirical', marker_color='#00d4aa', opacity=0.6,
))
fig.add_trace(go.Scatter(
    x=x_grid, y=t_pdf, mode='lines',
    name=f'Standardised t(ν={nu_production:.2f})',
    line=dict(color='white', width=2),
))
fig.add_vline(x=threshold_99, line_dash='dash', line_color='#E24B4A',
              annotation_text='1% threshold')
fig.add_vline(x=-threshold_99, line_dash='dash', line_color='#E24B4A')

fig.update_layout(
    template='plotly_dark',
    title='Standardised residuals vs production Student t distribution',
    xaxis_title='Standardised residual',
    yaxis_title='Density', height=400,
)
fig.show()
```




```python
# Annual anomaly counts. The final year runs only to the data cutoff, so it is
# excluded from the year-on-year comparison and reported separately below.
anomaly_series = pd.Series(anomaly_99.values, index=returns.index, name='anomaly')
annual_all = anomaly_series.groupby(anomaly_series.index.year).sum()

cutoff_year = returns.index.max().year
complete = annual_all[annual_all.index < cutoff_year]
partial_count = int(annual_all.get(cutoff_year, 0))
partial_days = int((anomaly_series.index.year == cutoff_year).sum())

fig = go.Figure(go.Bar(
    x=complete.index.astype(str), y=complete.values,
    marker_color='#E24B4A',
))
fig.add_hline(
    y=252 * 0.01, line_dash='dash', line_color='white', opacity=0.5,
    annotation_text='Expected (~2.5/year)',
)
fig.update_layout(
    template='plotly_dark',
    title=f'Anomaly flags by year (2000 to {cutoff_year - 1})',
    xaxis_title='Year', yaxis_title='Anomaly count',
    height=400,
)
fig.show()

display(Markdown(f"""
The chart shows complete calendar years only. {cutoff_year} is excluded because
it runs to the data cutoff of {returns.index.max().date()}, covering
{partial_days} trading days, and its {partial_count} flag{'s' if partial_count != 1 else ''}
are not comparable to a full year against the {252 * 0.01:.1f}-per-year line.
"""))
```





The chart shows complete calendar years only. 2026 is excluded because
it runs to the data cutoff of 2026-09-25, covering
184 trading days, and its 2 flags
are not comparable to a full year against the 2.5-per-year line.



## 12. Limitations

The anomaly flags are computed from a full-sample GARCH fit, meaning
the model uses future observations to estimate its parameters. In a
production system, anomaly detection would run on a walk-forward basis:
each day's standardised residual would be computed from a model fitted
only on past data. Full-sample residuals are used here for the
descriptive validation because the purpose is to characterise the
anomaly landscape, not to simulate real-time detection.

The threshold is calibrated to the model's fitted distribution. If the
model is misspecified (and no model is perfectly specified), the
theoretical tail probabilities may not match empirical rates exactly.
The excess anomaly ratio in the calibration table above quantifies this
gap.

This notebook uses a single anomaly detection method. Alternative
approaches (Isolation Forest, rolling distributional tests, Mahalanobis
distance on the feature set) could complement the GARCH-residual method
but are outside the scope of Version 1.

## 13. Conclusion


```python
display(Markdown(f"""
The GJR-GARCH standardised residuals provide a natural anomaly signal. At the
1% threshold, {n_99} of {n_total:,} trading days were flagged, and the exact
binomial coverage test does not reject the nominal rate
(p = {cov_99.pvalue:.4f}).

Two independent lenses describe where these flags fall. By calendar window,
{in_events} of {n_99} sit inside the three known stress periods, close to the
{expected_in_events:.1f} an even spread would place there, so the flags do not
concentrate in recognised crises. By volatility regime, {quiet_anomalies} fall
in Calm or Normal conditions. These counts partition the same flags on
different axes and are not meant to reconcile.

Together they fix the flag's role. It is not a crisis detector; the regime
classifier identifies stress. The anomaly flag marks days that surprised the
model relative to its own conditional variance, which is the question the
regime label cannot answer: not "is volatility high?" but "did something happen
the model did not expect?"

The flag and z-score export to Notebook 08, where they join the volatility
forecast and regime label in the Daily Market Risk Report.
"""))
```



The GJR-GARCH standardised residuals provide a natural anomaly signal. At the
1% threshold, 56 of 6,721 trading days were flagged, and the exact
binomial coverage test does not reject the nominal rate
(p = 0.1776).

Two independent lenses describe where these flags fall. By calendar window,
3 of 56 sit inside the three known stress periods, close to the
2.7 an even spread would place there, so the flags do not
concentrate in recognised crises. By volatility regime, 49 fall
in Calm or Normal conditions. These counts partition the same flags on
different axes and are not meant to reconcile.

Together they fix the flag's role. It is not a crisis detector; the regime
classifier identifies stress. The anomaly flag marks days that surprised the
model relative to its own conditional variance, which is the question the
regime label cannot answer: not "is volatility high?" but "did something happen
the model did not expect?"

The flag and z-score export to Notebook 08, where they join the volatility
forecast and regime label in the Daily Market Risk Report.



## 14. Export


```python
# ── Save anomaly data for NB08 ──
export_df = pd.DataFrame({
    'log_returns': returns.values,
    'z_garch': std_resids.values,
    'anomaly_1pct': anomaly_99.values,
    'anomaly_5pct': anomaly_95.values,
    'cond_vol_ann': cond_vol_ann.values,
    'regime': regimes.values,
}, index=returns.index)

export_path = Path('../data/nb07_anomalies.parquet')
export_df.to_parquet(export_path)
print(f'Anomaly data saved to {export_path.resolve()}')
print(f'Shape: {export_df.shape}')

# ── Export metrics to locked_metrics.json ──
metrics['notebook_07'] = {
    'nu_used_for_threshold': float(nu_production),
    'nu_source':             'notebook_05.garch_nu',
    'nu_refit_local':        round(float(nu), 2),
    'nu_refit_local_nobs':   int(n_total),
    'threshold_99':          round(float(threshold_99), 3),
    'threshold_95':          round(float(threshold_95), 3),
    'n_anomalies_1pct':      int(n_99),
    'n_anomalies_5pct':      int(n_95),
    'n_total_days':          int(n_total),
    'coverage_p_1pct':       float(cov_99.pvalue),
    'coverage_p_5pct':       float(cov_95.pvalue),
    'pct_in_known_events':   round(float(pct_in_events), 1),
    'quiet_period_anomalies': int(quiet_anomalies),
    'neg_anomaly_share':     round(float(neg_anomalies / n_99 * 100), 1),
}

metrics_path.write_text(json.dumps(metrics, indent=2))
print(f'\nExported notebook_07 metrics to {metrics_path.resolve()}')
for k, v in metrics['notebook_07'].items():
    print(f'  {k}: {v}')
```

    Anomaly data saved to C:\Users\Mena\Documents\Python\sp500-market-intelligence\data\nb07_anomalies.parquet
    Shape: (6721, 6)
    
    Exported notebook_07 metrics to C:\Users\Mena\Documents\Python\sp500-market-intelligence\data\locked_metrics.json
      nu_used_for_threshold: 6.760109587620447
      nu_source: notebook_05.garch_nu
      nu_refit_local: 6.94
      nu_refit_local_nobs: 6721
      threshold_99: 2.972
      threshold_95: 1.999
      n_anomalies_1pct: 56
      n_anomalies_5pct: 327
      n_total_days: 6721
      coverage_p_1pct: 0.17756784429545253
      coverage_p_5pct: 0.6342710221416591
      pct_in_known_events: 5.4
      quiet_period_anomalies: 49
      neg_anomaly_share: 87.5
    
