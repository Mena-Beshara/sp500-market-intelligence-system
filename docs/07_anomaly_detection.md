# 07 — Anomaly detection

Identifying market observations that the selected volatility model did not
expect.

The regime classifier in Notebook 05 groups market conditions into four
states (Calm, Normal, Stress, Crisis) using percentile thresholds on
conditional volatility. It answers "what kind of market is this" on a
scale calibrated to historical experience. But percentile thresholds
cannot distinguish between a day that is merely in the 96th percentile
and a day that falls outside the historical distribution entirely.

This notebook adds a complementary layer: anomaly detection using the selected
model's own standardised residuals. If the model is well specified, these
residuals follow the innovation distribution Notebook 05 selected, at its fitted
shape parameters. Days that land in either tail of that distribution are events
the model did not anticipate, even after accounting for time-varying volatility,
the larger variance response to falls, fat tails and skew. These are the days a
risk manager most needs to know about.

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


```python
# Refit the specification NB05 selected on the full cleaned history
nb05 = metrics['notebook_05']
best_label = nb05['best_label']
garch_dist = nb05['garch_dist']

# The GJR orders are typed below, so stop if NB05 selected another model.
assert best_label.startswith('GJR-GARCH(1,1,1)'), (
    f'NB05 selected {best_label}; update the orders in this refit')

# arch library uses percentage returns
returns_pct = returns * 100

am = arch_model(returns_pct, vol='GARCH', p=1, o=1, q=1, dist=garch_dist)
res = am.fit(disp='off')

# Shape parameters by the distribution's own names, checked against NB05's
# export so no value is paired with the wrong parameter.
shape_names = am.distribution.parameter_names()
assert set(shape_names) == set(nb05['garch_shape']), (
    f'Refit parameters {shape_names} do not match notebook_05.garch_shape')

# NB05's label carries an em dash; the prose uses 'with' instead.
model_name = best_label.replace(' \u2014 ', ' with ')

display(Markdown(f"""
The anomaly detector uses the standardised residuals of the model selected in
Notebook 05, {model_name}. The model is refit here on all {res.nobs:,} daily
returns in the cleaned file, so every trading day gets a residual. The
production fit in Notebook 05 used {nb05['garch_nobs']:,} returns from the
feature file, which starts after the warm-up rows Notebook 03 drops.
"""))

print(res.summary())
```



The anomaly detector uses the standardised residuals of the model selected in
Notebook 05, GJR-GARCH(1,1,1) with skewed Student's t. The model is refit here on all 6,721 daily
returns in the cleaned file, so every trading day gets a residual. The
production fit in Notebook 05 used 6,441 returns from the
feature file, which starts after the warm-up rows Notebook 03 drops.



                             Constant Mean - GJR-GARCH Model Results                         
    =========================================================================================
    Dep. Variable:                       log_returns   R-squared:                       0.000
    Mean Model:                        Constant Mean   Adj. R-squared:                  0.000
    Vol Model:                             GJR-GARCH   Log-Likelihood:               -8911.98
    Distribution:      Standardized Skew Student's t   AIC:                           17838.0
    Method:                       Maximum Likelihood   BIC:                           17885.7
                                                       No. Observations:                 6721
    Date:                           Tue, Nov 10 2026   Df Residuals:                     6720
    Time:                                   22:58:41   Df Model:                            1
                                     Mean Model                                 
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    mu             0.0252  9.535e-03      2.639  8.305e-03 [6.479e-03,4.386e-02]
                                   Volatility Model                              
    =============================================================================
                     coef    std err          t      P>|t|       95.0% Conf. Int.
    -----------------------------------------------------------------------------
    omega          0.0199  3.236e-03      6.156  7.452e-10  [1.358e-02,2.626e-02]
    alpha[1]       0.0000  8.700e-03      0.000      1.000 [-1.705e-02,1.705e-02]
    gamma[1]       0.2108  2.102e-02     10.031  1.119e-23      [  0.170,  0.252]
    beta[1]        0.8798  1.252e-02     70.247      0.000      [  0.855,  0.904]
                                  Distribution                              
    ========================================================================
                     coef    std err          t      P>|t|  95.0% Conf. Int.
    ------------------------------------------------------------------------
    eta            7.6693      0.717     10.695  1.080e-26 [  6.264,  9.075]
    lambda        -0.1487  1.591e-02     -9.342  9.461e-21 [ -0.180, -0.117]
    ========================================================================
    
    Covariance estimator: robust
    


```python
# Standardised residuals and conditional volatility, then the refit's shape
# parameters beside NB05's production values
std_resids = res.std_resid
cond_vol = res.conditional_volatility / 100  # back to decimal
cond_vol_ann = cond_vol * np.sqrt(252)
shape_local = res.params[shape_names]

for name in shape_names:
    print(f'{name}: local refit {shape_local[name]:.4f}, '
          f'NB05 production {nb05["garch_shape"][name]:.4f}')
print(f'Standardised residuals: {len(std_resids):,} observations')
print(f'Mean: {std_resids.mean():.4f}, Std: {std_resids.std():.4f}')
```

    eta: local refit 7.6693, NB05 production 7.4861
    lambda: local refit -0.1487, NB05 production -0.1496
    Standardised residuals: 6,721 observations
    Mean: -0.0082, Std: 1.0052
    

## 5. Anomaly threshold


```python
# Cutoffs come from the distribution of the NB05 production fit, not from the
# refit above, which exists only to produce residuals across the full history.
# arch standardises residuals to unit variance and its ppf returns quantiles on
# that same scale, so no rescaling is needed. A skewed distribution needs its
# own cutoff in each tail.
shape_production = np.array([nb05['garch_shape'][name] for name in shape_names])
shape_text = ', '.join(f'{name} = {value:.4f}'
                       for name, value in zip(shape_names, shape_production))
dist = am.distribution

cut_99 = dist.ppf(np.array([0.005, 0.995]), shape_production)  # lower, upper
cut_95 = dist.ppf(np.array([0.025, 0.975]), shape_production)

# Flag anomalies tail by tail
low_99 = std_resids < cut_99[0]
high_99 = std_resids > cut_99[1]
low_95 = std_resids < cut_95[0]
high_95 = std_resids > cut_95[1]
anomaly_99 = low_99 | high_99
anomaly_95 = low_95 | high_95
z_abs = np.abs(std_resids)  # section 7 still ranks days by size

n_total = len(std_resids)
n_99 = int(anomaly_99.sum())
n_95 = int(anomaly_95.sum())

# Exact two-sided binomial tests: each tail against half the nominal rate,
# then both tails together against the full rate.
def tail_test(mask, share):
    """Observed count, expected count and two-sided binomial p-value."""
    observed = int(mask.sum())
    test = binomtest(observed, n_total, share, alternative='two-sided')
    return observed, n_total * share, test.pvalue

def p_value(p):
    """For tables: 0.1234, or < 0.0001 when too small to show."""
    return '< 0.0001' if p < 1e-4 else f'{p:.4f}'

def p_text(p):
    """For prose: p = 0.1234, or p < 0.0001."""
    return 'p < 0.0001' if p < 1e-4 else f'p = {p:.4f}'

calibration = [
    ('5% (warning)', 'lower', f'{cut_95[0]:+.3f}', *tail_test(low_95, 0.025)),
    ('5% (warning)', 'upper', f'{cut_95[1]:+.3f}', *tail_test(high_95, 0.025)),
    ('5% (warning)', 'both', '', *tail_test(anomaly_95, 0.05)),
    ('1% (anomaly)', 'lower', f'{cut_99[0]:+.3f}', *tail_test(low_99, 0.005)),
    ('1% (anomaly)', 'upper', f'{cut_99[1]:+.3f}', *tail_test(high_99, 0.005)),
    ('1% (anomaly)', 'both', '', *tail_test(anomaly_99, 0.01)),
]

# Pooled tests, kept for the conclusion and the export
cov_99 = binomtest(n_99, n_total, 0.01, alternative='two-sided')
cov_95 = binomtest(n_95, n_total, 0.05, alternative='two-sided')

# Split test, as in NB05 section 5.5: of the flagged days, the share below the
# lower cutoff, tested against one half. Each tail holds the same share of days
# under the fitted distribution, so the flags should split evenly whatever
# their total.
def split_test(low, flags):
    """Flags below the lower cutoff, all flags, and the two-sided binomial
    p-value of that count against one half."""
    n_low = int(low.sum())
    n_flags = int(flags.sum())
    test = binomtest(n_low, n_flags, 0.5, alternative='two-sided')
    return n_low, n_flags, test.pvalue

split_95 = split_test(low_95, anomaly_95)  # kept for the export
split_99 = split_test(low_99, anomaly_99)
split_results = [('5% (warning)', *split_95), ('1% (anomaly)', *split_99)]

table_rows = '\n'.join(
    f'| {level} | {tail} | {cut} | {expected:.1f} | {observed} | '
    f'{observed / expected:.2f}× | {p_value(p)} |'
    for level, tail, cut, observed, expected, p in calibration)

split_rows = '\n'.join(
    f'| {level} | {n_flags} | {n_low} | {n_low / n_flags:.1%} | {p_value(p)} |'
    for level, n_low, n_flags, p in split_results)

# Rows whose p-value, the last item, is below 0.05
rejected = [row for row in calibration if row[-1] < 0.05]
if not rejected:
    verdict = ('No test rejects correct coverage at the 5% significance level, so '
               "neither tail's flag count, nor the total, departs significantly "
               'from the fitted distribution.')
else:
    parts = [f"the {level.split()[0]} {'tails together' if tail == 'both' else tail + ' tail'} "
             f"({observed} flags against {expected:.1f} expected, {p_text(p)})"
             for level, tail, cut, observed, expected, p in rejected]
    verdict = ('Coverage is rejected at the 5% significance level for '
               + '; '.join(parts)
               + '. A rejected row means the fitted distribution misstates how often '
               'residuals pass that cutoff, so each side concerned needs review '
               'before the flag feeds the risk report.')

def split_sentence(level, n_low, n_flags, p):
    """One sentence per level: the split, its p-value and what it shows."""
    text = (f'At the {level.split()[0]} level, {n_low} of {n_flags} flags '
            f'({n_low / n_flags:.1%}) fall below the lower cutoff ({p_text(p)})')
    if p >= 0.05:
        return text + ', consistent with an even split.'
    side = 'falling' if n_low > n_flags / 2 else 'rising'
    return text + f', so the flagged days lean to the {side} side.'

split_verdict = ' '.join(split_sentence(*row) for row in split_results)
if any(row[-1] < 0.05 for row in split_results):
    split_verdict += (' A lean means the fitted distribution misplaces the balance '
                      'between falls and rises at that level, even where each '
                      "tail's count passes on its own.")

display(Markdown(f"""
| Level | Tail | Cutoff | Expected | Observed | Ratio | Coverage p |
|---|---|---|---|---|---|---|
{table_rows}

Each tail has its own cutoff, from the distribution of the production fit in
Notebook 05 ({model_name}: {shape_text}, {nb05['garch_nobs']:,} observations).
arch standardises residuals to unit variance and its quantile function works on
that scale, so the cutoffs need no rescaling. A day is flagged when its
standardised residual falls below the lower cutoff or above the upper one.

Coverage is tested with exact two-sided binomial tests: each tail against half
the nominal rate, and both tails together against the full rate. {verdict}

| Level | Flags | Below lower cutoff | Share below | Split p |
|---|---|---|---|---|
{split_rows}

The per-tail tests look at each tail on its own. The split test from Notebook 05
section 5.5 compares the two: each tail holds the same share of days under the
fitted distribution, so the flags should divide evenly between them whatever
their total. {split_verdict}

The 1% level is the primary anomaly flag for the Daily Market Risk Report, and
the 5% level is available as a warning.
"""))
```



| Level | Tail | Cutoff | Expected | Observed | Ratio | Coverage p |
|---|---|---|---|---|---|---|
| 5% (warning) | lower | -2.143 | 168.0 | 170 | 1.01× | 0.8758 |
| 5% (warning) | upper | +1.836 | 168.0 | 156 | 0.93× | 0.3688 |
| 5% (warning) | both |  | 336.1 | 326 | 0.97× | 0.5949 |
| 1% (anomaly) | lower | -3.214 | 33.6 | 43 | 1.28× | 0.1183 |
| 1% (anomaly) | upper | +2.618 | 33.6 | 24 | 0.71× | 0.1002 |
| 1% (anomaly) | both |  | 67.2 | 67 | 1.00× | 1.0000 |

Each tail has its own cutoff, from the distribution of the production fit in
Notebook 05 (GJR-GARCH(1,1,1) with skewed Student's t: eta = 7.4861, lambda = -0.1496, 6,441 observations).
arch standardises residuals to unit variance and its quantile function works on
that scale, so the cutoffs need no rescaling. A day is flagged when its
standardised residual falls below the lower cutoff or above the upper one.

Coverage is tested with exact two-sided binomial tests: each tail against half
the nominal rate, and both tails together against the full rate. No test rejects correct coverage at the 5% significance level, so neither tail's flag count, nor the total, departs significantly from the fitted distribution.

| Level | Flags | Below lower cutoff | Share below | Split p |
|---|---|---|---|---|
| 5% (warning) | 326 | 170 | 52.1% | 0.4716 |
| 1% (anomaly) | 67 | 43 | 64.2% | 0.0271 |

The per-tail tests look at each tail on its own. The split test from Notebook 05
section 5.5 compares the two: each tail holds the same share of days under the
fitted distribution, so the flags should divide evenly between them whatever
their total. At the 5% level, 170 of 326 flags (52.1%) fall below the lower cutoff (p = 0.4716), consistent with an even split. At the 1% level, 43 of 67 flags (64.2%) fall below the lower cutoff (p = 0.0271), so the flagged days lean to the falling side. A lean means the fitted distribution misplaces the balance between falls and rises at that level, even where each tail's count passes on its own.

The 1% level is the primary anomaly flag for the Daily Market Risk Report, and
the 5% level is available as a warning.



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
# Each day's tail probability under the production fit's distribution: the
# chance of a surprise at least this far out on the same side. Ranking by it
# puts falls and rises on the same scale as the cutoffs in section 5.
cdf_values = dist.cdf(np.asarray(std_resids), shape_production)
tail_prob = np.where(std_resids.values < 0, cdf_values, 1 - cdf_values)

anomaly_df = pd.DataFrame({
    'date': returns.index,
    'log_return': returns.values,
    'z_score': std_resids.values,
    'abs_z': z_abs.values,
    'tail_prob': tail_prob,
    'cond_vol_ann': cond_vol_ann.values,
}).set_index('date')

rarest = anomaly_df.nsmallest(20, 'tail_prob')
top_20 = rarest[['log_return', 'z_score', 'tail_prob', 'cond_vol_ann']].copy()
top_20.columns = ['Log return', 'z score', 'About 1 day in', 'Ann. cond. vol']
top_20['Log return'] = (top_20['Log return'] * 100).round(2).astype(str) + '%'
top_20['z score'] = top_20['z score'].round(2)
top_20['About 1 day in'] = (1 / top_20['About 1 day in']).round().astype(int).map('{:,}'.format)
top_20['Ann. cond. vol'] = (top_20['Ann. cond. vol'] * 100).round(1).astype(str) + '%'

display(top_20)
print(f'\nRises among the 20 rarest days: {int((rarest["z_score"] > 0).sum())}')

# Asymmetry check
neg_anomalies = (std_resids[anomaly_99] < 0).sum()
pos_anomalies = (std_resids[anomaly_99] > 0).sum()
print(f'Negative anomalies: {neg_anomalies} ({neg_anomalies/n_99*100:.0f}%)')
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
      <th>z score</th>
      <th>About 1 day in</th>
      <th>Ann. cond. vol</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-03</th>
      <td>-3.58%</td>
      <td>-7.36</td>
      <td>22,545</td>
      <td>7.8%</td>
    </tr>
    <tr>
      <th>2020-06-11</th>
      <td>-6.08%</td>
      <td>-7.17</td>
      <td>18,933</td>
      <td>13.5%</td>
    </tr>
    <tr>
      <th>2007-02-27</th>
      <td>-3.53%</td>
      <td>-6.82</td>
      <td>13,681</td>
      <td>8.3%</td>
    </tr>
    <tr>
      <th>2016-06-24</th>
      <td>-3.66%</td>
      <td>-6.33</td>
      <td>8,532</td>
      <td>9.2%</td>
    </tr>
    <tr>
      <th>2018-10-10</th>
      <td>-3.34%</td>
      <td>-5.58</td>
      <td>3,902</td>
      <td>9.6%</td>
    </tr>
    <tr>
      <th>2025-10-10</th>
      <td>-2.75%</td>
      <td>-5.44</td>
      <td>3,341</td>
      <td>8.1%</td>
    </tr>
    <tr>
      <th>2016-09-09</th>
      <td>-2.48%</td>
      <td>-5.17</td>
      <td>2,472</td>
      <td>7.7%</td>
    </tr>
    <tr>
      <th>2024-12-18</th>
      <td>-2.99%</td>
      <td>-5.04</td>
      <td>2,144</td>
      <td>9.5%</td>
    </tr>
    <tr>
      <th>2021-11-26</th>
      <td>-2.3%</td>
      <td>-4.49</td>
      <td>1,107</td>
      <td>8.2%</td>
    </tr>
    <tr>
      <th>2026-06-05</th>
      <td>-2.68%</td>
      <td>-4.39</td>
      <td>972</td>
      <td>9.8%</td>
    </tr>
    <tr>
      <th>2021-01-27</th>
      <td>-2.6%</td>
      <td>-4.32</td>
      <td>892</td>
      <td>9.7%</td>
    </tr>
    <tr>
      <th>2022-11-10</th>
      <td>5.4%</td>
      <td>3.40</td>
      <td>861</td>
      <td>25.1%</td>
    </tr>
    <tr>
      <th>2012-09-06</th>
      <td>2.02%</td>
      <td>3.34</td>
      <td>785</td>
      <td>9.5%</td>
    </tr>
    <tr>
      <th>2020-02-24</th>
      <td>-3.41%</td>
      <td>-4.18</td>
      <td>747</td>
      <td>13.0%</td>
    </tr>
    <tr>
      <th>2025-04-03</th>
      <td>-4.96%</td>
      <td>-4.17</td>
      <td>739</td>
      <td>19.0%</td>
    </tr>
    <tr>
      <th>2009-03-23</th>
      <td>6.84%</td>
      <td>3.27</td>
      <td>690</td>
      <td>33.0%</td>
    </tr>
    <tr>
      <th>2017-05-17</th>
      <td>-1.83%</td>
      <td>-4.12</td>
      <td>689</td>
      <td>7.2%</td>
    </tr>
    <tr>
      <th>2010-11-04</th>
      <td>1.91%</td>
      <td>3.22</td>
      <td>628</td>
      <td>9.3%</td>
    </tr>
    <tr>
      <th>2016-11-07</th>
      <td>2.2%</td>
      <td>3.16</td>
      <td>559</td>
      <td>10.9%</td>
    </tr>
    <tr>
      <th>2001-05-16</th>
      <td>2.81%</td>
      <td>3.13</td>
      <td>534</td>
      <td>14.1%</td>
    </tr>
  </tbody>
</table>
</div>


    
    Rises among the 20 rarest days: 6
    Negative anomalies: 43 (64%)
    Positive anomalies: 24 (36%)
    


```python
# The sign split of the 1% flags, read against section 5's split test.
neg_share = neg_anomalies / n_99
lean = split_sentence(*split_results[1])

display(Markdown(f"""
Of the {n_99} flags at the 1% level, {neg_anomalies} are falls and
{pos_anomalies} are rises ({neg_share:.1%} falls). Each 1% cutoff should catch
the same number of days under the fitted distribution, so an even split is
expected. {lean}

The GJR term does not drive this split. Gamma raises the next day's variance
after a fall, which changes the size of later residuals, not the sign of
today's. Once each return is divided by its conditional volatility, the balance
between falls and rises comes from the innovation distribution alone, so a lean
here points at the shape of the fitted distribution, not at the variance
equation.

The largest flags are examined against known events in the next section.
Section 8 tests whether they concentrate in crisis windows; the answer shapes
how the flag should be read.
"""))
```



Of the 67 flags at the 1% level, 43 are falls and
24 are rises (64.2% falls). Each 1% cutoff should catch
the same number of days under the fitted distribution, so an even split is
expected. At the 1% level, 43 of 67 flags (64.2%) fall below the lower cutoff (p = 0.0271), so the flagged days lean to the falling side.

The GJR term does not drive this split. Gamma raises the next day's variance
after a fall, which changes the size of later residuals, not the sign of
today's. Once each return is divided by its conditional volatility, the balance
between falls and rises comes from the innovation distribution alone, so a lean
here points at the shape of the fitted distribution, not at the variance
equation.

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
display(val_df.set_index('Event'))
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
      <th>Window</th>
      <th>Trading days</th>
      <th>Anomalies</th>
      <th>Rate</th>
    </tr>
    <tr>
      <th>Event</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Global financial crisis</th>
      <td>2008-09-01 to 2009-03-31</td>
      <td>146</td>
      <td>2</td>
      <td>1.4%</td>
    </tr>
    <tr>
      <th>COVID crash</th>
      <td>2020-02-15 to 2020-04-30</td>
      <td>52</td>
      <td>1</td>
      <td>1.9%</td>
    </tr>
    <tr>
      <th>Rate-hike onset</th>
      <td>2022-01-01 to 2022-06-30</td>
      <td>124</td>
      <td>0</td>
      <td>0.0%</td>
    </tr>
    <tr>
      <th>Full sample (baseline)</th>
      <td>2000-01-04 to 2026-09-25</td>
      <td>6721</td>
      <td>67</td>
      <td>1.0%</td>
    </tr>
  </tbody>
</table>
</div>



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



**3** of **67** anomaly flags (4.5%) fall within
the three known stress windows. Those windows cover **322** trading
days, **4.8%** of the sample, so an even spread through time would
place **3.2** flags inside them. An exact two-sided
binomial test of the observed count against that share gives
p = **1.0000**.

Flags fall inside the stress windows at a rate consistent with their share of calendar time. This is the expected behaviour of a well-specified volatility model: GARCH raises its conditional variance during sustained stress, so crisis-period moves become less surprising once the model has adjusted, and standardised residuals should not cluster in crises. The anomaly flag is therefore not a crisis detector. Crisis identification is the regime classifier's role; the anomaly flag marks days that surprised the model relative to its own volatility estimate.



## 9. Anomaly and regime interaction


```python
# Regime labels on NB08's rule: a day exactly on a cut-off belongs to the
# higher regime (Calm below the 25th percentile, Normal from the 25th up to
# the 75th, Stress from the 75th up to the 95th, Crisis from the 95th).
q25, q75, q95 = cond_vol_ann.quantile([0.25, 0.75, 0.95])

regimes = pd.cut(
    cond_vol_ann,
    bins=[-np.inf, q25, q75, q95, np.inf],
    labels=['Calm', 'Normal', 'Stress', 'Crisis'],
    right=False,
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
        'Days': int(total),
        'Anomalies': int(flagged),
        'Rate': f'{flagged/total*100:.1f}%' if total > 0 else 'n/a',
    })

regime_cross_df = pd.DataFrame(regime_anomaly)
display(regime_cross_df.set_index('Regime'))
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
      <th>Days</th>
      <th>Anomalies</th>
      <th>Rate</th>
    </tr>
    <tr>
      <th>Regime</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Calm</th>
      <td>1680</td>
      <td>30</td>
      <td>1.8%</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>3360</td>
      <td>28</td>
      <td>0.8%</td>
    </tr>
    <tr>
      <th>Stress</th>
      <td>1344</td>
      <td>7</td>
      <td>0.5%</td>
    </tr>
    <tr>
      <th>Crisis</th>
      <td>337</td>
      <td>2</td>
      <td>0.6%</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Flags per regime against the 1% the model promises, and one test of whether
# the rate is the same in every regime.
from scipy.stats import chi2_contingency

calm_anomalies = int(regime_cross_df.loc[
    regime_cross_df['Regime'] == 'Calm', 'Anomalies'
].iloc[0])
normal_anomalies = int(regime_cross_df.loc[
    regime_cross_df['Regime'] == 'Normal', 'Anomalies'
].iloc[0])
quiet_anomalies = calm_anomalies + normal_anomalies

regime_rows = []
for _, row in regime_cross_df.iterrows():
    days, flags = int(row['Days']), int(row['Anomalies'])
    p = binomtest(flags, days, 0.01, alternative='two-sided').pvalue
    regime_rows.append((row['Regime'], days, flags, days * 0.01, p))

chi2_stat, regime_p, regime_dof, _ = chi2_contingency(
    [[flags, days - flags] for _, days, flags, _, _ in regime_rows])

regime_table = '\n'.join(
    f'| {name} | {days:,} | {flags} | {expected:.1f} | {flags / expected:.2f}× | {p_value(p)} |'
    for name, days, flags, expected, p in regime_rows)

over = [name for name, days, flags, expected, p in regime_rows if p < 0.05 and flags > expected]
under = [name for name, days, flags, expected, p in regime_rows if p < 0.05 and flags < expected]

if regime_p < 0.05:
    rate_text = (f'The flag rate is not the same in every regime (chi-square = '
                 f'{chi2_stat:.2f}, {regime_dof} degrees of freedom, {p_text(regime_p)}).')
    if over:
        rate_text += f" {' and '.join(over)} days set off the flag more often than the 1% the model promises."
    if under:
        rate_text += f" {' and '.join(under)} days set off the flag less often than the 1% the model promises."
    rate_text += (' Dividing each return by its conditional volatility should leave a '
                  'well-specified model with a 1% rate in every regime, so the gap points at '
                  'a part of the model that fits some regimes worse than others. The '
                  f'{quiet_anomalies} flags in Calm or Normal regimes therefore partly reflect '
                  'that misfit, not only events the regime label misses.')
else:
    rate_text = (f'The flag rate is consistent with being the same in every regime '
                 f'(chi-square = {chi2_stat:.2f}, {regime_dof} degrees of freedom, '
                 f'{p_text(regime_p)}), as a well-specified model implies. The '
                 f'{quiet_anomalies} flags in Calm or Normal regimes are days the regime '
                 'label alone would not mark.')

display(Markdown(f"""
| Regime | Days | Flags | Expected at 1% | Ratio | Binomial p |
|---|---|---|---|---|---|
{regime_table}

{rate_text}
"""))
```



| Regime | Days | Flags | Expected at 1% | Ratio | Binomial p |
|---|---|---|---|---|---|
| Calm | 1,680 | 30 | 16.8 | 1.79× | 0.0029 |
| Normal | 3,360 | 28 | 33.6 | 0.83× | 0.3852 |
| Stress | 1,344 | 7 | 13.4 | 0.52× | 0.0971 |
| Crisis | 337 | 2 | 3.4 | 0.59× | 0.7797 |

The flag rate is not the same in every regime (chi-square = 15.14, 3 degrees of freedom, p = 0.0017). Calm days set off the flag more often than the 1% the model promises. Dividing each return by its conditional volatility should leave a well-specified model with a 1% rate in every regime, so the gap points at a part of the model that fits some regimes worse than others. The 58 flags in Calm or Normal regimes therefore partly reflect that misfit, not only events the regime label misses.



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
    opacity=0.5, annotation_text=f'Expected ({63 * 0.01:.2f} per quarter)',
)
fig.update_layout(
    template='plotly_dark',
    title='Rolling quarterly anomaly count',
    xaxis_title='Date', yaxis_title='Anomalies per 63 days',
    height=400,
)
fig.show()

# Do flags cluster in time? Ljung-Box on the daily 0/1 flag series: under a
# well-specified model the flags are independent, so no autocorrelation.
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_lags = 21
lb = acorr_ljungbox(anomaly_99.astype(int), lags=[lb_lags], return_df=True)
lb_stat, lb_p = float(lb['lb_stat'].iloc[0]), float(lb['lb_pvalue'].iloc[0])
peak = int(rolling_count.max())
peak_end = rolling_count.idxmax().date()

if lb_p < 0.05:
    cluster_text = ('the flags cluster in time more than independent draws would, so '
                    'one surprise makes another more likely within weeks')
else:
    cluster_text = ('the flags show no significant clustering, as independent '
                    'residuals from a well-specified model imply')

events_text = ('more often than' if cov_events.pvalue < 0.05 and in_events > expected_in_events
               else 'less often than' if cov_events.pvalue < 0.05
               else 'at about the rate of')
regime_text = ('differs across' if regime_p < 0.05 else 'does not differ significantly across')

display(Markdown(f"""
The rolling count peaks at {peak} flags in the 63 days ending {peak_end}, against
{63 * 0.01:.2f} expected per quarter. A Ljung-Box test on the daily flag series
at {lb_lags} lags gives Q = {lb_stat:.2f} ({p_text(lb_p)}): {cluster_text}.

Where flags fall is a separate question, answered earlier. Section 8 found them
inside the three known stress windows {events_text} those windows' share of the
calendar ({in_events} of {n_99} flags, p = {cov_events.pvalue:.4f}), and section 9
found that the flag rate {regime_text} volatility regimes. A rise in the rolling
count is therefore read against those results, not as a crisis signal in itself.
"""))
```





The rolling count peaks at 4 flags in the 63 days ending 2024-02-22, against
0.63 expected per quarter. A Ljung-Box test on the daily flag series
at 21 lags gives Q = 14.81 (p = 0.8322): the flags show no significant clustering, as independent residuals from a well-specified model imply.

Where flags fall is a separate question, answered earlier. Section 8 found them
inside the three known stress windows at about the rate of those windows' share of the
calendar (3 of 67 flags, p = 1.0000), and section 9
found that the flag rate differs across volatility regimes. A rise in the rolling
count is therefore read against those results, not as a crisis signal in itself.



## 11. Residual distribution


```python
# Compare empirical residuals with the production fit's distribution. arch
# gives the log density at unit variance, so exp() of it is the density on the
# same scale as the residuals. Same distribution and cutoffs as section 5.
x_grid = np.linspace(-8, 8, 500)
fitted_pdf = np.exp(dist.loglikelihood(shape_production, x_grid,
                                       np.ones_like(x_grid), individual=True))

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=std_resids.values, nbinsx=100, histnorm='probability density',
    name='Empirical', marker_color='#00d4aa', opacity=0.6,
))
fig.add_trace(go.Scatter(
    x=x_grid, y=fitted_pdf, mode='lines',
    name=f'Fitted distribution ({shape_text})',
    line=dict(color='white', width=2),
))
fig.add_vline(x=cut_99[0], line_dash='dash', line_color='#E24B4A',
              annotation_text=f'1% lower cutoff ({cut_99[0]:+.3f})',
              annotation_position='top left')
fig.add_vline(x=cut_99[1], line_dash='dash', line_color='#E24B4A',
              annotation_text=f'1% upper cutoff ({cut_99[1]:+.3f})',
              annotation_position='top right')

fig.update_layout(
    template='plotly_dark',
    title="Standardised residuals vs the production fit's distribution",
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

This notebook uses a single anomaly detection method. Alternative
approaches (Isolation Forest, rolling distributional tests, Mahalanobis
distance on the feature set) could complement the GARCH-residual method
but are outside the scope of Version 1.


```python
# Limitations rendered from the calibration checks in sections 5 and 9
lean = [(level.split()[0], n_low, n_flags, p)
        for level, n_low, n_flags, p in split_results if p < 0.05]

if rejected:
    coverage_summary = (f'{len(rejected)} of the {len(calibration)} coverage tests in '
                        'section 5 rejected the nominal rate at the 5% level')
else:
    coverage_summary = (f'none of the {len(calibration)} coverage tests in section 5 '
                        'rejected the nominal rate at the 5% level')

if lean:
    lean_summary = 'the split test found the flags leaning at the ' + ' and '.join(
        f'{lvl} level ({n_low} of {n_flags} below the lower cutoff, {p_text(p)})'
        for lvl, n_low, n_flags, p in lean)
else:
    lean_summary = 'the split test found no lean at either level'

regime_summary = (f'the flag rate differs across volatility regimes ({p_text(regime_p)})'
                  if regime_p < 0.05 else
                  f'the flag rate does not differ significantly across volatility '
                  f'regimes ({p_text(regime_p)})')

if rejected or lean or regime_p < 0.05:
    gap_text = ('The nominal 1% and 5% rates are therefore approximate. Two explanations '
                'fit these results but are not tested here: one skew parameter shapes both '
                'the body and the extreme tails of the fitted distribution, and one '
                'distribution serves every volatility regime.')
else:
    gap_text = 'The nominal rates are consistent with the data on every check run here.'

display(Markdown(f"""
The cutoffs assume the fitted distribution describes the residuals. Checked
against the data, {coverage_summary}; {lean_summary}; and section 9 found that
{regime_summary}. {gap_text}

The residuals come from this notebook's refit on {res.nobs:,} days, while the
cutoffs use the production fit's shape from {nb05['garch_nobs']:,} days. Section 4
prints both sets of shape parameters side by side.
"""))
```



The cutoffs assume the fitted distribution describes the residuals. Checked
against the data, none of the 6 coverage tests in section 5 rejected the nominal rate at the 5% level; the split test found the flags leaning at the 1% level (43 of 67 below the lower cutoff, p = 0.0271); and section 9 found that
the flag rate differs across volatility regimes (p = 0.0017). The nominal 1% and 5% rates are therefore approximate. Two explanations fit these results but are not tested here: one skew parameter shapes both the body and the extreme tails of the fitted distribution, and one distribution serves every volatility regime.

The residuals come from this notebook's refit on 6,721 days, while the
cutoffs use the production fit's shape from 6,441 days. Section 4
prints both sets of shape parameters side by side.



## 13. Conclusion


```python
# Conclusion rendered from the results of sections 5 to 10
pooled_text = 'does not reject' if cov_99.pvalue >= 0.05 else 'rejects'
lean_line = split_sentence(*split_results[1])

if cov_events.pvalue < 0.05 and in_events > expected_in_events:
    role_text = ('Flags concentrate in the known stress windows, so the flag partly '
                 'tracks crises as well as surprises.')
else:
    role_text = ('Flags do not concentrate in the known stress windows, so the flag is '
                 'not a crisis detector; identifying stress is the regime classifier\'s '
                 'job. The flag marks days that surprised the model relative to its own '
                 'conditional variance.')
if regime_p < 0.05:
    role_text += (' Because the flag rate differs across regimes, part of what it '
                  'marks is where the model fits less well, so a flag reads as a '
                  'surprise to this model rather than to the market at large.')

display(Markdown(f"""
At the 1% level, {n_99} of {n_total:,} trading days were flagged against
{n_total * 0.01:.1f} expected, and the pooled coverage test {pooled_text} the
nominal rate ({p_text(cov_99.pvalue)}). {lean_line}

Three checks describe where and when the flags fall. By calendar window,
{in_events} of {n_99} sit inside the three known stress periods against
{expected_in_events:.1f} for an even spread ({p_text(cov_events.pvalue)}), so they
fall there {events_text} those windows' share of the calendar. By volatility
regime, {quiet_anomalies} fall in Calm or Normal conditions, and the flag rate
{regime_text} regimes ({p_text(regime_p)}). Over time, {cluster_text} (Ljung-Box
at {lb_lags} lags, {p_text(lb_p)}).

{role_text}

The flag and z-score export to Notebook 08, where they join the volatility
forecast and regime label in the Daily Market Risk Report.
"""))
```



At the 1% level, 67 of 6,721 trading days were flagged against
67.2 expected, and the pooled coverage test does not reject the
nominal rate (p = 1.0000). At the 1% level, 43 of 67 flags (64.2%) fall below the lower cutoff (p = 0.0271), so the flagged days lean to the falling side.

Three checks describe where and when the flags fall. By calendar window,
3 of 67 sit inside the three known stress periods against
3.2 for an even spread (p = 1.0000), so they
fall there at about the rate of those windows' share of the calendar. By volatility
regime, 58 fall in Calm or Normal conditions, and the flag rate
differs across regimes (p = 0.0017). Over time, the flags show no significant clustering, as independent residuals from a well-specified model imply (Ljung-Box
at 21 lags, p = 0.8322).

Flags do not concentrate in the known stress windows, so the flag is not a crisis detector; identifying stress is the regime classifier's job. The flag marks days that surprised the model relative to its own conditional variance. Because the flag rate differs across regimes, part of what it marks is where the model fits less well, so a flag reads as a surprise to this model rather than to the market at large.

The flag and z-score export to Notebook 08, where they join the volatility
forecast and regime label in the Daily Market Risk Report.



## 14. Export


```python
# Save anomaly data for NB08
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

# Export metrics to locked_metrics.json. The whole block is replaced, so no
# key from the symmetric version survives.
coverage_p = {(level.split()[0], tail): p
              for level, tail, cut, observed, expected, p in calibration}

metrics['notebook_07'] = {
    'dist_used_for_cutoffs':  garch_dist,
    'shape_used_for_cutoffs': {name: float(value)
                               for name, value in zip(shape_names, shape_production)},
    'shape_source':           'notebook_05.garch_shape',
    'shape_refit_local':      {name: round(float(shape_local[name]), 4)
                               for name in shape_names},
    'refit_local_nobs':       int(res.nobs),
    'cutoff_99_lower':        round(float(cut_99[0]), 3),
    'cutoff_99_upper':        round(float(cut_99[1]), 3),
    'cutoff_95_lower':        round(float(cut_95[0]), 3),
    'cutoff_95_upper':        round(float(cut_95[1]), 3),
    'n_anomalies_1pct':       int(n_99),
    'n_anomalies_5pct':       int(n_95),
    'n_lower_1pct':           int(low_99.sum()),
    'n_upper_1pct':           int(high_99.sum()),
    'n_lower_5pct':           int(low_95.sum()),
    'n_upper_5pct':           int(high_95.sum()),
    'n_total_days':           int(n_total),
    'coverage_p_1pct':        float(cov_99.pvalue),
    'coverage_p_5pct':        float(cov_95.pvalue),
    'coverage_p_1pct_lower':  float(coverage_p[('1%', 'lower')]),
    'coverage_p_1pct_upper':  float(coverage_p[('1%', 'upper')]),
    'coverage_p_5pct_lower':  float(coverage_p[('5%', 'lower')]),
    'coverage_p_5pct_upper':  float(coverage_p[('5%', 'upper')]),
    'split_p_1pct':           float(split_99[2]),
    'split_p_5pct':           float(split_95[2]),
    'pct_in_known_events':    round(float(pct_in_events), 1),
    'quiet_period_anomalies': int(quiet_anomalies),
    'neg_anomaly_share':      round(float(neg_anomalies / n_99 * 100), 1),
}

metrics_path.write_text(json.dumps(metrics, indent=2))
print(f'\nExported notebook_07 metrics to {metrics_path.resolve()}')
for k, v in metrics['notebook_07'].items():
    print(f'  {k}: {v}')
```

    Anomaly data saved to C:\Users\Mena\Documents\Python\sp500-market-intelligence\data\nb07_anomalies.parquet
    Shape: (6721, 6)
    
    Exported notebook_07 metrics to C:\Users\Mena\Documents\Python\sp500-market-intelligence\data\locked_metrics.json
      dist_used_for_cutoffs: skewt
      shape_used_for_cutoffs: {'eta': 7.4860859169257195, 'lambda': -0.14955043971231133}
      shape_source: notebook_05.garch_shape
      shape_refit_local: {'eta': 7.6693, 'lambda': -0.1487}
      refit_local_nobs: 6721
      cutoff_99_lower: -3.214
      cutoff_99_upper: 2.618
      cutoff_95_lower: -2.143
      cutoff_95_upper: 1.836
      n_anomalies_1pct: 67
      n_anomalies_5pct: 326
      n_lower_1pct: 43
      n_upper_1pct: 24
      n_lower_5pct: 170
      n_upper_5pct: 156
      n_total_days: 6721
      coverage_p_1pct: 1.0
      coverage_p_5pct: 0.5949351391695523
      coverage_p_1pct_lower: 0.11831878867776251
      coverage_p_1pct_upper: 0.1001936549763022
      coverage_p_5pct_lower: 0.8757758666094415
      coverage_p_5pct_upper: 0.36883319105816115
      split_p_1pct: 0.027119992403126333
      split_p_5pct: 0.47157797712896615
      pct_in_known_events: 4.5
      quiet_period_anomalies: 58
      neg_anomaly_share: 64.2
    
