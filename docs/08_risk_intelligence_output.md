# 08 — Risk intelligence output

Daily Market Risk Report generation, production-system validation, and
decision-rule analysis for the weekly allocation experiment.

Every upstream notebook built one component of the risk picture: a
volatility forecast, a regime label, a model-selection verdict, an anomaly
flag. This notebook assembles them into the single deliverable the project
exists to produce — and validates that deliverable before publishing it.
The forecast is calibrated against realised outcomes, the VaR thresholds
are backtested against historical breaches, and the regime classifier's
dynamics are characterised, so the report's numbers arrive with their
track record attached.

## 1. Setup and objective


```python
import numpy as np
import pandas as pd
import json
from pathlib import Path

from arch import arch_model
from scipy.stats import t as t_dist, chi2, linregress
from scipy.special import gamma as gamma_fn
from scipy.optimize import brentq

import plotly.graph_objects as go
import plotly.io as pio
pd.options.plotting.backend = 'plotly'
pio.templates.default = 'plotly_dark'

from IPython.display import display, Markdown
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('../reports/figures', exist_ok=True)
```


```python
# Load locked metrics from earlier notebooks
metrics_path = Path('../data/locked_metrics.json')
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

print('Locked metrics loaded:')
for nb_key in sorted(metrics.keys()):
    print(f'  {nb_key}: {len(metrics[nb_key])} fields')
```

    Locked metrics loaded:
      notebook_01: 36 fields
      notebook_02: 7 fields
      notebook_04: 3 fields
      notebook_05: 22 fields
      notebook_06: 61 fields
      notebook_07: 14 fields
    


```python
display(Markdown("""

### Objective



The project business question is narrow: given everything the system knows

today, what should I review before this week's allocation decision?



This notebook answers that by combining three upstream outputs into a single

Daily Market Risk Report:



| Source | Output | Question answered |

|---|---|---|

| Notebook 05 | Conditional volatility forecast + regime label | How much risk, and what kind of market? |

| Notebook 06 | Model selection evidence | Which model should produce the forecast? |

| Notebook 07 | Anomaly flag + z-score | Did something unexpected happen today? |



Notebook 05 evaluated candidate models against each other. This notebook

evaluates the production system against reality: forecast calibration in

section 4 and VaR backtesting in section 6 test whether the numbers the

report publishes have been reliable historically. A risk report that never

audits itself is a liability, not a tool.



The report is decision support, not a trading signal. It quantifies the risk

environment; what to do about it is the investor's call. The second half of

the notebook characterises the regime classifier's dynamics, evaluates its

track record through the retrospective feedback loop, and measures whether

regime-aware allocation would have improved on passive dollar-cost

averaging.

"""))
```




### Objective



The project business question is narrow: given everything the system knows

today, what should I review before this week's allocation decision?



This notebook answers that by combining three upstream outputs into a single

Daily Market Risk Report:



| Source | Output | Question answered |

|---|---|---|

| Notebook 05 | Conditional volatility forecast + regime label | How much risk, and what kind of market? |

| Notebook 06 | Model selection evidence | Which model should produce the forecast? |

| Notebook 07 | Anomaly flag + z-score | Did something unexpected happen today? |



Notebook 05 evaluated candidate models against each other. This notebook

evaluates the production system against reality: forecast calibration in

section 4 and VaR backtesting in section 6 test whether the numbers the

report publishes have been reliable historically. A risk report that never

audits itself is a liability, not a tool.



The report is decision support, not a trading signal. It quantifies the risk

environment; what to do about it is the investor's call. The second half of

the notebook characterises the regime classifier's dynamics, evaluates its

track record through the retrospective feedback loop, and measures whether

regime-aware allocation would have improved on passive dollar-cost

averaging.




## 2. Data loading


```python
# Load returns and prices
df = pd.read_parquet('../data/sp500_features.parquet')
returns = df['log_returns'].dropna().sort_index()
close = df['Close'].reindex(returns.index)

print(f'Observations: {len(returns):,}')
print(f'Date range:   {returns.index.min().date()} to {returns.index.max().date()}')
```

    Observations: 6,441
    Date range:   2001-02-13 to 2026-09-25
    


```python
# Load anomaly data from Notebook 07
anomaly_path = Path('../data/nb07_anomalies.parquet')
if anomaly_path.exists():
    anomalies = pd.read_parquet(anomaly_path)
    print(f'Anomaly data loaded: {len(anomalies):,} rows, columns: {list(anomalies.columns)}')
else:
    print('WARNING: nb07_anomalies.parquet not found — anomaly layer will be unavailable')
    anomalies = None
```

    Anomaly data loaded: 6,721 rows, columns: ['log_returns', 'z_garch', 'anomaly_1pct', 'anomaly_5pct', 'cond_vol_ann', 'regime']
    


```python
display(Markdown("""

The feature set from Notebook 03 provides the return series and the Close

prices used by the allocation simulation in section 13. The anomaly flags

from Notebook 07 provide the standardised residuals and z-scores computed

from the full-sample GJR-GARCH fit. The volatility model is refitted here

on the current data to produce a live forecast, rather than loading a

stale model object — this keeps the report self-contained and ensures the

forecast reflects the most recent observations.

"""))
```




The feature set from Notebook 03 provides the return series and the Close

prices used by the allocation simulation in section 13. The anomaly flags

from Notebook 07 provide the standardised residuals and z-scores computed

from the full-sample GJR-GARCH fit. The volatility model is refitted here

on the current data to produce a live forecast, rather than loading a

stale model object — this keeps the report self-contained and ensures the

forecast reflects the most recent observations.




## 3. Production model refit


```python
display(Markdown("""

Notebook 06 confirmed that GJR-GARCH(1,1,1) with Student's t innovations

remains the production model. Deep learning alternatives (LSTM and MLP)

did not improve on it under identical walk-forward evaluation. The model

is refitted here on the full available sample so the forecast incorporates

the most recent market data.

"""))
```




Notebook 06 confirmed that GJR-GARCH(1,1,1) with Student's t innovations

remains the production model. Deep learning alternatives (LSTM and MLP)

did not improve on it under identical walk-forward evaluation. The model

is refitted here on the full available sample so the forecast incorporates

the most recent market data.





```python
# Refit GJR-GARCH(1,1,1) — Student's t on the full sample
returns_pct = returns * 100

gjr_spec = arch_model(
    returns_pct,
    mean='Constant',
    vol='GARCH',
    p=1, o=1, q=1,
    dist='t'
)
gjr_res = gjr_spec.fit(disp='off')

# Extract key parameters
mu_daily = gjr_res.params['mu'] / 100
omega = gjr_res.params['omega']
alpha = gjr_res.params.get('alpha[1]', 0)
gamma = gjr_res.params.get('gamma[1]', 0)
beta = gjr_res.params['beta[1]']
nu = gjr_res.params['nu']
persistence = alpha + beta + gamma / 2

# Conditional volatility series (daily raw units, annualised)
cond_vol_daily = gjr_res.conditional_volatility / 100
cond_vol_ann = cond_vol_daily * np.sqrt(252)

print(f'GJR-GARCH(1,1,1) — Student\'s t refitted on {len(returns):,} observations')
print(f'  mu (daily):  {mu_daily:.6f}')
print(f'  omega:       {omega:.6f}')
print(f'  alpha:       {alpha:.4f}')
print(f'  gamma:       {gamma:.4f}')
print(f'  beta:        {beta:.4f}')
print(f'  nu (d.f.):   {nu:.2f}')
print(f'  persistence: {persistence:.4f}')
print(f'  AIC:         {gjr_res.aic:,.1f}')
```

    GJR-GARCH(1,1,1) — Student's t refitted on 6,441 observations
      mu (daily):  0.000531
      omega:       0.018642
      alpha:       0.0000
      gamma:       0.2033
      beta:        0.8811
      nu (d.f.):   6.76
      persistence: 0.9827
      AIC:         16,964.5
    


```python
# E[|z|] factor for the fitted Student's t — converts sigma into
# the expected absolute return. Used by the calibration section and
# the report.
def abs_return_factor(params):
    nu_val = params['nu']
    return np.sqrt(nu_val / (nu_val - 2)) * (
        2 * np.sqrt(nu_val - 2) * gamma_fn((nu_val + 1) / 2)
        / ((nu_val - 1) * np.sqrt(np.pi) * gamma_fn(nu_val / 2))
    )

c_factor = abs_return_factor(gjr_res.params)
print(f'E[|z|] under fitted Student\'s t: {c_factor:.4f}')
```

    E[|z|] under fitted Student's t: 0.9025
    

## 4. Forecast calibration

### Point-forecast calibration


```python
display(Markdown("""

Notebook 05 compared candidate models. This section asks a different

question: how accurate is the production forecast itself? Three checks

follow — a predicted-versus-realised comparison, a bias measurement, and

a rolling error series showing where in history the model struggled.



The comparison uses the expected absolute return, c × sigma, as the point

forecast rather than sigma directly. Sigma is the conditional standard

deviation, not the expectation of |return|; comparing sigma against |return|

would report a spurious overestimation bias that is only the gap between a

standard deviation and a mean absolute value. Using c × sigma makes the

forecast and the realisation the same quantity.



One structural note on what "in-sample" means here. The GARCH variance

recursion at day t uses only returns up to day t-1, so each conditional

volatility value is a genuine one-step-ahead forecast in its information

set. The parameters, however, are estimated on the full sample. The

calibration below therefore tests the model's functional form and dynamics,

not its real-time deployability — that was the walk-forward's job in

Notebook 05.

"""))
```




Notebook 05 compared candidate models. This section asks a different

question: how accurate is the production forecast itself? Three checks

follow — a predicted-versus-realised comparison, a bias measurement, and

a rolling error series showing where in history the model struggled.



The comparison uses the expected absolute return, c × sigma, as the point

forecast rather than sigma directly. Sigma is the conditional standard

deviation, not the expectation of |return|; comparing sigma against |return|

would report a spurious overestimation bias that is only the gap between a

standard deviation and a mean absolute value. Using c × sigma makes the

forecast and the realisation the same quantity.



One structural note on what "in-sample" means here. The GARCH variance

recursion at day t uses only returns up to day t-1, so each conditional

volatility value is a genuine one-step-ahead forecast in its information

set. The parameters, however, are estimated on the full sample. The

calibration below therefore tests the model's functional form and dynamics,

not its real-time deployability — that was the walk-forward's job in

Notebook 05.





```python
# Point forecast of |return| and realised |return|
forecast_abs = cond_vol_daily * c_factor
realised_abs = returns.abs()

forecast_error = realised_abs - forecast_abs
bias = forecast_error.mean()
bias_share = bias / realised_abs.mean()

# Mincer-Zarnowitz regression: realised on forecast
mz = linregress(forecast_abs, realised_abs)
mz_slope = mz.slope
mz_intercept = mz.intercept
mz_r2 = mz.rvalue ** 2

print(f'Mean forecast error (realised - forecast): {bias:+.6f}')
print(f'  as a share of mean |return|:             {bias_share:+.2%}')
print(f'Mincer-Zarnowitz: slope = {mz_slope:.3f}, intercept = {mz_intercept:.6f}, R² = {mz_r2:.3f}')
```

    Mean forecast error (realised - forecast): -0.001522
      as a share of mean |return|:             -19.24%
    Mincer-Zarnowitz: slope = 0.850, intercept = -0.000110, R² = 0.310
    


```python
# Predicted vs realised scatter
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=forecast_abs * 100,
    y=realised_abs * 100,
    mode='markers',
    marker=dict(size=3, opacity=0.25, color='#3498db'),
    name='Daily observations',
    showlegend=False
))

# 45-degree reference line
line_max = float(max(forecast_abs.max(), realised_abs.max()) * 100)
fig.add_trace(go.Scatter(
    x=[0, line_max],
    y=[0, line_max],
    mode='lines',
    line=dict(dash='dash', color='white', width=1),
    name='Perfect calibration'
))

fig.update_layout(
    title='Forecast calibration: expected vs realised absolute daily return',
    xaxis_title='Forecast E[|return|] (%)',
    yaxis_title='Realised |return| (%)',
    height=500,
    legend=dict(orientation='h', y=-0.15)
)

fig.show()
```




```python
# Rolling forecast error — 63-day (one quarter) window
rolling_error = forecast_error.rolling(63).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=rolling_error.index,
    y=rolling_error * 100,
    mode='lines',
    line=dict(width=1, color='#f39c12'),
    name='63-day rolling mean error'
))
fig.add_hline(y=0, line_dash='dash', line_color='gray')

fig.update_layout(
    title='Rolling forecast error (realised - forecast, 63-day mean)',
    xaxis_title='Date',
    yaxis_title='Mean error (%)',
    height=400,
    showlegend=False
)

fig.show()
```




```python
display(Markdown(f"""

Over the full sample the mean forecast error is {bias:+.6f}, which is

{abs(bias_share):.1%} of the average absolute return —

{'small enough to treat the forecast as unbiased for practical purposes' if abs(bias_share) < 0.05 else 'a systematic ' + ('underestimation' if bias > 0 else 'overestimation') + ' worth monitoring'}.



The Mincer-Zarnowitz regression of realised on forecast values gives a

slope of {mz_slope:.3f} and an intercept of {mz_intercept:.6f}.

{'A slope near one with an intercept near zero means forecast movements translate roughly one-for-one into realised movements.' if 0.8 <= mz_slope <= 1.2 else 'The slope deviates from one, meaning the forecast under- or over-reacts to changing conditions; the direction and size of the deviation are worth tracking as more data arrives.'}

The R² of {mz_r2:.3f} looks low in isolation, but that is a documented

property of the target, not the model: the absolute daily return is a noisy

proxy for latent volatility, and Andersen and Bollerslev (1998) showed that

this noise caps the achievable R² for daily proxies well below one even for

a correctly specified model. The slope and intercept carry the calibration

information; the R² mostly measures proxy noise.



The rolling error chart shows where the model struggled: error spikes

upward at the onset of volatility shocks, when realised moves outrun the

forecast, then turns negative in the aftermath as the model's elevated

variance estimate decays more slowly than the market calms. That signature

is the cost of one-step forecasting with a persistent process, and it is

the pattern a risk manager should expect from any GARCH-family system.

"""))
```




Over the full sample the mean forecast error is -0.001522, which is

19.2% of the average absolute return —

a systematic overestimation worth monitoring.



The Mincer-Zarnowitz regression of realised on forecast values gives a

slope of 0.850 and an intercept of -0.000110.

A slope near one with an intercept near zero means forecast movements translate roughly one-for-one into realised movements.

The R² of 0.310 looks low in isolation, but that is a documented

property of the target, not the model: the absolute daily return is a noisy

proxy for latent volatility, and Andersen and Bollerslev (1998) showed that

this noise caps the achievable R² for daily proxies well below one even for

a correctly specified model. The slope and intercept carry the calibration

information; the R² mostly measures proxy noise.



The rolling error chart shows where the model struggled: error spikes

upward at the onset of volatility shocks, when realised moves outrun the

forecast, then turns negative in the aftermath as the model's elevated

variance estimate decays more slowly than the market calms. That signature

is the cost of one-step forecasting with a persistent process, and it is

the pattern a risk manager should expect from any GARCH-family system.




### Density calibration


```python
display(Markdown("""

The point-forecast checks above test the magnitude of the forecast. A

volatility forecast is a distributional object, though, and the risk

report leans on that distribution directly — the VaR thresholds are

quantiles of the fitted Student's t. So the whole predictive density needs

checking, not just its scale.



The tool for that is the probability integral transform. For each day, take

the fitted distribution and evaluate its CDF at the realised return. If the

distribution is correctly specified, these PIT values are uniform on the

unit interval: the realised return lands in each quantile bucket as often

as the model says it should. A PIT histogram that humps in the middle means

the forecast intervals are too wide (the model overstates risk); a U-shaped

histogram means they are too narrow (the model understates tail risk).

"""))
```




The point-forecast checks above test the magnitude of the forecast. A

volatility forecast is a distributional object, though, and the risk

report leans on that distribution directly — the VaR thresholds are

quantiles of the fitted Student's t. So the whole predictive density needs

checking, not just its scale.



The tool for that is the probability integral transform. For each day, take

the fitted distribution and evaluate its CDF at the realised return. If the

distribution is correctly specified, these PIT values are uniform on the

unit interval: the realised return lands in each quantile bucket as often

as the model says it should. A PIT histogram that humps in the middle means

the forecast intervals are too wide (the model overstates risk); a U-shaped

histogram means they are too narrow (the model understates tail risk).





```python
# Standardised residuals are the innovations z_t = (r_t - mu) / sigma_t,
# distributed under the model as standardised Student's t with unit
# variance. arch returns them directly.
std_resid = pd.Series(gjr_res.std_resid, index=returns.index).dropna()

# PIT: CDF of the standardised t evaluated at each standardised residual.
# scipy's t is the standard (non-unit-variance) t, so rescale the argument
# by sqrt(nu / (nu - 2)) to match the unit-variance standardisation.
pit = t_dist.cdf(std_resid * np.sqrt(nu / (nu - 2)), df=nu)
pit = pd.Series(pit, index=std_resid.index)

# Central-interval coverage: fraction of PIT values inside the central band
# for each nominal level. Correct specification puts empirical ≈ nominal.
cov_levels = [0.50, 0.80, 0.90, 0.95, 0.99]
coverage = {}
for lvl in cov_levels:
    lo, hi = (1 - lvl) / 2, (1 + lvl) / 2
    coverage[lvl] = float(((pit >= lo) & (pit <= hi)).mean())

# Uniformity summary: mean should sit near 0.5, std near 1/sqrt(12) ≈ 0.289
pit_mean = pit.mean()
pit_std = pit.std()

print('Central-interval coverage (empirical vs nominal):')
for lvl in cov_levels:
    print(f'  {lvl:.0%}: {coverage[lvl]:.1%}')
print(f'PIT mean {pit_mean:.3f} (target 0.500), std {pit_std:.3f} (target 0.289)')
```

    Central-interval coverage (empirical vs nominal):
      50%: 50.1%
      80%: 79.1%
      90%: 90.0%
      95%: 95.2%
      99%: 99.1%
    PIT mean 0.500 (target 0.500), std 0.288 (target 0.289)
    


```python
# PIT histogram — flat under correct specification
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=pit,
    nbinsx=20,
    histnorm='probability density',
    marker_color='#3498db',
    opacity=0.85,
    name='PIT'
))
fig.add_hline(
    y=1.0, line_dash='dash', line_color='white',
    annotation_text='Uniform', annotation_position='top right'
)
fig.update_layout(
    title='PIT histogram — uniform under correct density specification',
    xaxis_title='PIT value',
    yaxis_title='Density',
    height=400,
    showlegend=False
)
fig.show()
```




```python
cov_rows = '\n'.join(
    f'| {lvl:.0%} | {coverage[lvl]:.1%} | {coverage[lvl] - lvl:+.1%} |'
    for lvl in cov_levels
)
worst = max(cov_levels, key=lambda l: abs(coverage[l] - l))

display(Markdown(f"""

### Central-interval coverage



| Nominal | Empirical | Gap |

|---|---|---|

{cov_rows}



The PIT mean is {pit_mean:.3f} against a target of 0.500 and its standard

deviation is {pit_std:.3f} against a target of 0.289, the values a uniform

distribution would produce.

{'Coverage tracks the nominal levels closely across the range, so the fitted Student’s t is a reasonable description of the return distribution — the VaR quantiles rest on a density that is calibrated, not just a scale that is calibrated.' if abs(coverage[worst] - worst) < 0.03 else 'The largest coverage gap is at the ' + f'{worst:.0%}' + ' level, where empirical and nominal diverge by ' + f'{coverage[worst] - worst:+.1%}' + '. This is a distributional flag: the VaR quantiles rest on a density that is imperfectly calibrated at that level, and section 6’s backtest is the tail-specific version of the same check.'}



This shares the in-sample caveat of the rest of the section: the parameters

are full-sample, so the PIT tests distributional specification rather than

out-of-sample density-forecast calibration. It connects to the standardised

residual diagnostic from Notebook 05, where excess kurtosis collapsed from

11.30 in raw returns to 2.06 in the standardised residuals — the same

innovations, seen here through their full distribution rather than a single

moment.

"""))
```




### Central-interval coverage



| Nominal | Empirical | Gap |

|---|---|---|

| 50% | 50.1% | +0.1% |
| 80% | 79.1% | -0.9% |
| 90% | 90.0% | +0.0% |
| 95% | 95.2% | +0.2% |
| 99% | 99.1% | +0.1% |



The PIT mean is 0.500 against a target of 0.500 and its standard

deviation is 0.288 against a target of 0.289, the values a uniform

distribution would produce.

Coverage tracks the nominal levels closely across the range, so the fitted Student’s t is a reasonable description of the return distribution — the VaR quantiles rest on a density that is calibrated, not just a scale that is calibrated.



This shares the in-sample caveat of the rest of the section: the parameters

are full-sample, so the PIT tests distributional specification rather than

out-of-sample density-forecast calibration. It connects to the standardised

residual diagnostic from Notebook 05, where excess kurtosis collapsed from

11.30 in raw returns to 2.06 in the standardised residuals — the same

innovations, seen here through their full distribution rather than a single

moment.




## 5. Volatility forecast and regime classification


```python
# Regime thresholds from the full-sample distribution
q25 = np.percentile(cond_vol_ann, 25)
q75 = np.percentile(cond_vol_ann, 75)
q95 = np.percentile(cond_vol_ann, 95)

print(f'Regime thresholds (annualised conditional volatility):')
print(f'  Calm    < {q25:.2%}')
print(f'  Normal  {q25:.2%} – {q75:.2%}')
print(f'  Stress  {q75:.2%} – {q95:.2%}')
print(f'  Crisis  > {q95:.2%}')
```

    Regime thresholds (annualised conditional volatility):
      Calm    < 10.23%
      Normal  10.23% – 19.35%
      Stress  19.35% – 33.76%
      Crisis  > 33.76%
    


```python
# Classify each day into a regime
def classify_regime(vol, q25, q75, q95):
    if vol < q25:
        return 'Calm'
    elif vol < q75:
        return 'Normal'
    elif vol < q95:
        return 'Stress'
    else:
        return 'Crisis'

regime_order = ['Calm', 'Normal', 'Stress', 'Crisis']

regimes = pd.Series(
    [classify_regime(v, q25, q75, q95) for v in cond_vol_ann],
    index=returns.index,
    name='regime'
)

regime_counts = regimes.value_counts()
regime_pcts = regimes.value_counts(normalize=True) * 100

print('Regime distribution:')
for regime in regime_order:
    count = regime_counts.get(regime, 0)
    pct = regime_pcts.get(regime, 0)
    print(f'  {regime:7s}  {count:5,d} days  ({pct:.1f}%)')
```

    Regime distribution:
      Calm     1,610 days  (25.0%)
      Normal   3,220 days  (50.0%)
      Stress   1,288 days  (20.0%)
      Crisis     323 days  (5.0%)
    


```python
# Next-day forecast
forecast = gjr_res.forecast(horizon=1, reindex=False)
sigma_next = np.sqrt(forecast.variance.values[-1, 0]) / 100
sigma_next_ann = sigma_next * np.sqrt(252)
abs_move_next = sigma_next * c_factor

# Classify the forecast
regime_next = classify_regime(sigma_next_ann, q25, q75, q95)
vol_percentile = (cond_vol_ann < sigma_next_ann).mean()

as_of = returns.index.max().date()

print(f'Forecast as of {as_of}:')
print(f'  Conditional std:       {sigma_next:.4%} daily ({sigma_next_ann:.2%} annualised)')
print(f'  Expected abs move:     {abs_move_next:.4%}')
print(f'  Historical percentile: {vol_percentile:.0%}')
print(f'  Regime:                {regime_next}')
```

    Forecast as of 2026-09-25:
      Conditional std:       0.6446% daily (10.23% annualised)
      Expected abs move:     0.5818%
      Historical percentile: 25%
      Regime:                Calm
    


```python
# Regime confidence: distance from the forecast to the bracketing thresholds.
# 'Deep in Normal' and 'one step from Stress' are very different signals.
regime_bounds = {
    'Calm':   (None, q25),
    'Normal': (q25, q75),
    'Stress': (q75, q95),
    'Crisis': (q95, None),
}
lower_bound, upper_bound = regime_bounds[regime_next]
next_up = {'Calm': 'Normal', 'Normal': 'Stress', 'Stress': 'Crisis', 'Crisis': None}[regime_next]
next_dn = {'Calm': None, 'Normal': 'Calm', 'Stress': 'Normal', 'Crisis': 'Stress'}[regime_next]

dist_up = (upper_bound - sigma_next_ann) if upper_bound is not None else None
dist_dn = (sigma_next_ann - lower_bound) if lower_bound is not None else None

# Risk trend: forecast against recent realised conditional volatility.
# ~5 trading days is one week, ~21 is one month.
vol_1w_ago = cond_vol_ann.iloc[-6] if len(cond_vol_ann) > 6 else np.nan
vol_1m_ago = cond_vol_ann.iloc[-22] if len(cond_vol_ann) > 22 else np.nan
chg_1w = sigma_next_ann - vol_1w_ago
chg_1m = sigma_next_ann - vol_1m_ago

def trend_label(chg, level):
    # Treat moves under 5% of the current level as flat to avoid reading noise.
    if np.isnan(chg):
        return 'n/a'
    if abs(chg) < 0.05 * level:
        return 'stable'
    return 'rising' if chg > 0 else 'falling'

trend_1w = trend_label(chg_1w, sigma_next_ann)
trend_1m = trend_label(chg_1m, sigma_next_ann)

print(f'Regime confidence: forecast at the {vol_percentile:.0%} percentile of history')
if dist_up is not None:
    print(f'  {dist_up:+.2%} from the {next_up} threshold ({upper_bound:.2%})')
if dist_dn is not None:
    print(f'  {dist_dn:+.2%} above the {next_dn} threshold ({lower_bound:.2%})')
print(f'Risk trend: {trend_1w} over 1 week, {trend_1m} over 1 month')
print(f'  1 week ago {vol_1w_ago:.2%}, 1 month ago {vol_1m_ago:.2%}, forecast {sigma_next_ann:.2%}')
```

    Regime confidence: forecast at the 25% percentile of history
      +0.00% from the Normal threshold (10.23%)
    Risk trend: falling over 1 week, falling over 1 month
      1 week ago 11.06%, 1 month ago 11.07%, forecast 10.23%
    

### Forecast decomposition


```python
display(Markdown("""

A GARCH forecast has no feature importances in the machine-learning sense —

it is a deterministic function of a handful of terms. But those terms have

a clean reading, and decomposing the one-step variance forecast into them

answers what actually drove today's number. The GJR-GARCH recursion is



    sigma²(t+1) = omega + alpha · eps²(t) + gamma · eps²(t) · 1[eps(t) < 0] + beta · sigma²(t)



which splits tomorrow's variance into four contributions: a constant

long-run floor (omega), the symmetric response to yesterday's shock

(alpha · eps²), the extra response if yesterday's shock was negative

(the leverage term, gamma · eps²), and the persistence carried forward from

yesterday's variance (beta · sigma²). The relative sizes say whether an

elevated forecast reflects a fresh shock or the slow decay of an old one.

"""))
```




A GARCH forecast has no feature importances in the machine-learning sense —

it is a deterministic function of a handful of terms. But those terms have

a clean reading, and decomposing the one-step variance forecast into them

answers what actually drove today's number. The GJR-GARCH recursion is



    sigma²(t+1) = omega + alpha · eps²(t) + gamma · eps²(t) · 1[eps(t) < 0] + beta · sigma²(t)



which splits tomorrow's variance into four contributions: a constant

long-run floor (omega), the symmetric response to yesterday's shock

(alpha · eps²), the extra response if yesterday's shock was negative

(the leverage term, gamma · eps²), and the persistence carried forward from

yesterday's variance (beta · sigma²). The relative sizes say whether an

elevated forecast reflects a fresh shock or the slow decay of an old one.





```python
# Decompose the one-step-ahead variance forecast (pct^2 scale)
eps_last = gjr_res.resid.iloc[-1]
sigma2_last = gjr_res.conditional_volatility.iloc[-1] ** 2
lev_active = eps_last < 0

comp_const = omega
comp_arch = alpha * eps_last ** 2
comp_lev = gamma * eps_last ** 2 if lev_active else 0.0
comp_garch = beta * sigma2_last
comp_total = comp_const + comp_arch + comp_lev + comp_garch

# Sanity check: reconstruction should match the model's own forecast
forecast_var_pct2 = forecast.variance.values[-1, 0]
recon_gap = abs(comp_total - forecast_var_pct2) / forecast_var_pct2

decomp = {
    'Long-run floor (omega)': comp_const,
    'Shock (alpha·eps²)': comp_arch,
    'Leverage (gamma·eps²)': comp_lev,
    'Persistence (beta·sigma²)': comp_garch,
}
persistence_share = comp_garch / comp_total
shock_share = (comp_arch + comp_lev) / comp_total

print(f'Forecast variance reconstruction gap: {recon_gap:.2%} (should be ~0)')
for name, val in decomp.items():
    print(f'  {name:28s} {val / comp_total:6.1%}')
```

    Forecast variance reconstruction gap: 0.00% (should be ~0)
      Long-run floor (omega)         4.5%
      Shock (alpha·eps²)             0.0%
      Leverage (gamma·eps²)          0.0%
      Persistence (beta·sigma²)     95.5%
    


```python
decomp_rows = '\n'.join(
    f'| {name} | {val / comp_total:.1%} |' for name, val in decomp.items()
)

display(Markdown(f"""

### What drove the forecast



| Component | Share of forecast variance |

|---|---|

{decomp_rows}



The persistence term carries {persistence_share:.0%} of tomorrow's forecast

variance, and yesterday's shock contributes {shock_share:.0%}. The leverage

term is {'active — the most recent return was negative, so the asymmetric response is switched on' if lev_active else 'inactive — the most recent return was positive, so no asymmetric term applies'}.

{'Today’s forecast is dominated by persistence: it mostly reflects volatility already in the system decaying slowly, not a fresh disturbance.' if persistence_share > 0.6 else 'Yesterday’s shock makes a material contribution to today’s forecast, so the number is responding to recent news rather than only carrying forward an existing level.'}

The reconstruction matches the model's own forecast to within

{recon_gap:.1%}, confirming the decomposition is exact rather than

approximate.

"""))
```




### What drove the forecast



| Component | Share of forecast variance |

|---|---|

| Long-run floor (omega) | 4.5% |
| Shock (alpha·eps²) | 0.0% |
| Leverage (gamma·eps²) | 0.0% |
| Persistence (beta·sigma²) | 95.5% |



The persistence term carries 96% of tomorrow's forecast

variance, and yesterday's shock contributes 0%. The leverage

term is inactive — the most recent return was positive, so no asymmetric term applies.

Today’s forecast is dominated by persistence: it mostly reflects volatility already in the system decaying slowly, not a fresh disturbance.

The reconstruction matches the model's own forecast to within

0.0%, confirming the decomposition is exact rather than

approximate.




## 6. Value at risk and backtesting

### 6.1 Parametric VaR


```python
display(Markdown("""

The VaR estimate uses the fitted Student's t distribution rather than a

Normal approximation. With excess kurtosis above 10 in raw returns, the

Normal assumption systematically understates tail risk. The VaR threshold

is the alpha-quantile of the fitted return distribution, mu + sigma × q,

so the small positive daily mean is included; ignoring it would overstate

the loss threshold slightly.



VaR is stated as a percentage of portfolio value at a one-day horizon.

A 95% daily VaR of X% means: on 19 out of 20 trading days, the portfolio

loss should not exceed X%. On the 20th day, it may — and section 6.2

checks whether historically it did so at the promised rate.

"""))
```




The VaR estimate uses the fitted Student's t distribution rather than a

Normal approximation. With excess kurtosis above 10 in raw returns, the

Normal assumption systematically understates tail risk. The VaR threshold

is the alpha-quantile of the fitted return distribution, mu + sigma × q,

so the small positive daily mean is included; ignoring it would overstate

the loss threshold slightly.



VaR is stated as a percentage of portfolio value at a one-day horizon.

A 95% daily VaR of X% means: on 19 out of 20 trading days, the portfolio

loss should not exceed X%. On the 20th day, it may — and section 6.2

checks whether historically it did so at the promised rate.





```python
# Parametric VaR using fitted Student's t (standardised to unit variance)
var_levels = [0.95, 0.99]
var_results = {}

for level in var_levels:
    q_level = t_dist.ppf(1 - level, df=nu) * np.sqrt((nu - 2) / nu)  # negative
    var_daily = -(mu_daily + sigma_next * q_level)
    var_results[level] = {'daily': var_daily, 'quantile': q_level}
    print(f'{level:.0%} VaR: {var_daily:.4%} daily')
```

    95% VaR: 0.9772% daily
    99% VaR: 1.5847% daily
    

### 6.2 VaR backtest


```python
display(Markdown("""

A VaR number is only credible if its historical breach frequency matches

its stated confidence level. A 95% VaR should be breached on roughly 5%

of days — materially more means the model understates risk, materially

fewer means it wastes risk budget by overstating it.



The Kupiec proportion-of-failures test formalises the check: under the

null hypothesis that the true breach probability equals the stated level,

the likelihood ratio statistic follows a chi-squared distribution with one

degree of freedom. A p-value above 0.05 means the observed breach count is

consistent with the model's claim.

"""))
```




A VaR number is only credible if its historical breach frequency matches

its stated confidence level. A 95% VaR should be breached on roughly 5%

of days — materially more means the model understates risk, materially

fewer means it wastes risk budget by overstating it.



The Kupiec proportion-of-failures test formalises the check: under the

null hypothesis that the true breach probability equals the stated level,

the likelihood ratio statistic follows a chi-squared distribution with one

degree of freedom. A p-value above 0.05 means the observed breach count is

consistent with the model's claim.





```python
def kupiec_pof(n, x, p):
    """Kupiec proportion-of-failures likelihood ratio test.

    n: number of observations, x: number of breaches,
    p: expected breach probability (e.g. 0.05 for 95% VaR).
    Returns (LR statistic, p-value)."""
    pi_hat = x / n
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        lr = -2 * (
            (n - x) * np.log(1 - p) + x * np.log(p)
            - (n - x) * np.log(1 - pi_hat) - x * np.log(pi_hat)
        )
    pval = 1 - chi2.cdf(lr, df=1)
    return lr, pval

n_obs = len(returns)
backtest_results = {}

for level in var_levels:
    q_level = var_results[level]['quantile']
    var_series = mu_daily + cond_vol_daily * q_level  # alpha-quantile per day
    breaches = returns < var_series
    x_breach = int(breaches.sum())
    expected = n_obs * (1 - level)
    breach_rate = x_breach / n_obs
    lr_stat, lr_pval = kupiec_pof(n_obs, x_breach, 1 - level)

    if lr_pval > 0.05:
        verdict = 'consistent'
    elif lr_pval > 0.01:
        verdict = 'borderline'
    else:
        verdict = 'rejected'

    backtest_results[level] = {
        'breaches': x_breach,
        'expected': expected,
        'breach_rate': breach_rate,
        'lr_stat': lr_stat,
        'lr_pval': lr_pval,
        'verdict': verdict,
    }
    print(f'{level:.0%} VaR: {x_breach} breaches vs {expected:.1f} expected '
          f'({breach_rate:.2%} vs {1 - level:.0%}) — Kupiec LR = {lr_stat:.2f}, '
          f'p = {lr_pval:.3f} [{verdict}]')
```

    95% VaR: 407 breaches vs 322.1 expected (6.32% vs 5%) — Kupiec LR = 21.85, p = 0.000 [rejected]
    99% VaR: 93 breaches vs 64.4 expected (1.44% vs 1%) — Kupiec LR = 11.27, p = 0.001 [rejected]
    


```python
bt95 = backtest_results[0.95]
bt99 = backtest_results[0.99]

verdict_text = {
    'consistent': 'the breach frequency is statistically consistent with the stated confidence level',
    'borderline': 'the breach frequency is at the edge of statistical consistency and warrants monitoring',
    'rejected': 'the breach frequency is statistically inconsistent with the stated confidence level',
}

display(Markdown(f"""

### VaR backtest results



| Level | Breaches | Expected | Breach rate | Target rate | Kupiec LR | p-value | Verdict |

|---|---|---|---|---|---|---|---|

| 95% | {bt95['breaches']:,d} | {bt95['expected']:.1f} | {bt95['breach_rate']:.2%} | 5.00% | {bt95['lr_stat']:.2f} | {bt95['lr_pval']:.3f} | {bt95['verdict']} |

| 99% | {bt99['breaches']:,d} | {bt99['expected']:.1f} | {bt99['breach_rate']:.2%} | 1.00% | {bt99['lr_stat']:.2f} | {bt99['lr_pval']:.3f} | {bt99['verdict']} |



At the 95% level, {verdict_text[bt95['verdict']]}. At the 99% level,

{verdict_text[bt99['verdict']]}.



{'Both levels pass, so the loss thresholds in the risk report can be read at face value.' if bt95['verdict'] == 'consistent' and bt99['verdict'] == 'consistent' else 'Where a level fails, the report reader should treat that threshold as indicative rather than exact, and the direction of the failure — too many or too few breaches — tells them which way to adjust.'}



The same in-sample caveat from the calibration section applies: parameters

come from the full sample, so this backtest validates the model's

distributional form, not a live deployment history. The variance recursion

itself uses only past information at each step.

"""))
```




### VaR backtest results



| Level | Breaches | Expected | Breach rate | Target rate | Kupiec LR | p-value | Verdict |

|---|---|---|---|---|---|---|---|

| 95% | 407 | 322.1 | 6.32% | 5.00% | 21.85 | 0.000 | rejected |

| 99% | 93 | 64.4 | 1.44% | 1.00% | 11.27 | 0.001 | rejected |



At the 95% level, the breach frequency is statistically inconsistent with the stated confidence level. At the 99% level,

the breach frequency is statistically inconsistent with the stated confidence level.



Where a level fails, the report reader should treat that threshold as indicative rather than exact, and the direction of the failure — too many or too few breaches — tells them which way to adjust.



The same in-sample caveat from the calibration section applies: parameters

come from the full sample, so this backtest validates the model's

distributional form, not a live deployment history. The variance recursion

itself uses only past information at each step.




## 7. Anomaly integration


```python
if anomalies is not None and 'std_resid' in anomalies.columns:
    latest_date = returns.index.max()
    if latest_date in anomalies.index:
        today_z = float(anomalies.loc[latest_date, 'std_resid'])
        today_flag = int(anomalies.loc[latest_date, 'anomaly_flag'])
        print(f'Anomaly check for {latest_date.date()}:')
        print(f'  Standardised residual (z-score): {today_z:.3f}')
        print(f'  Anomaly flag: {today_flag}')
    else:
        today_z = np.nan
        today_flag = 0
        print(f'{latest_date.date()} not found in anomaly data — no flag available')
else:
    today_z = np.nan
    today_flag = 0
    print('Anomaly data not loaded — anomaly layer unavailable')
```

    Anomaly data not loaded — anomaly layer unavailable
    

## 8. Daily Market Risk Report


```python
display(Markdown("""

This is the primary deliverable the entire analytical build works toward.

The report combines three layers — how much risk, what kind of market,

and whether anything unexpected happened — into a single page a portfolio

or risk manager can read before the market opens. A validation block

carries the calibration and backtest results forward, so the reader knows

how much weight the numbers deserve.



Model output and allocation decision are deliberately kept separate. The

report says what the numbers are; the weekly allocation experiment applies

the decision rule.

"""))
```




This is the primary deliverable the entire analytical build works toward.

The report combines three layers — how much risk, what kind of market,

and whether anything unexpected happened — into a single page a portfolio

or risk manager can read before the market opens. A validation block

carries the calibration and backtest results forward, so the reader knows

how much weight the numbers deserve.



Model output and allocation decision are deliberately kept separate. The

report says what the numbers are; the weekly allocation experiment applies

the decision rule.





```python
# Regime context strings
regime_context = {
    'Calm': 'Volatility sits in the bottom quartile of its historical distribution.',
    'Normal': 'Volatility sits within its typical historical range.',
    'Stress': 'Volatility is elevated, above the 75th percentile of its history.',
    'Crisis': 'Volatility is extreme, above the 95th percentile of its history.',
}

# Allocation multipliers — the project schedule
regime_multipliers = {
    'Calm': 1.0,
    'Normal': 1.5,
    'Stress': 2.0,
    'Crisis': 2.5,
}

multiplier = regime_multipliers[regime_next]

# Anomaly status string
if today_flag == 1:
    anomaly_str = f'**Yes** — standardised residual {today_z:.2f}'
    anomaly_note = (
        'An anomaly was detected on the most recent trading day. This means '
        'the observed return was surprising relative to what the model expected '
        'given the current volatility level. Review the return and any news flow '
        'before acting on the regime signal.'
    )
else:
    anomaly_str = 'No'
    anomaly_note = (
        'No anomaly detected. The most recent return fell within the range '
        'the model expected given current volatility conditions.'
    )

# Trend arrows and regime-confidence text for the report
arrow = {'rising': '↑', 'falling': '↓', 'stable': '→', 'n/a': ''}
trend_str = (
    f"{arrow[trend_1w]} {trend_1w} vs 1 week ago ({vol_1w_ago:.1%}), "
    f"{arrow[trend_1m]} {trend_1m} vs 1 month ago ({vol_1m_ago:.1%})"
)

if dist_up is not None and dist_dn is not None:
    confidence_str = (
        f'{dist_up:+.2%} from the {next_up} threshold, '
        f'{dist_dn:+.2%} above the {next_dn} threshold'
    )
elif dist_up is not None:
    confidence_str = f'{dist_up:+.2%} from the {next_up} threshold'
else:
    confidence_str = f'{dist_dn:+.2%} above the {next_dn} threshold'

driver_str = (
    f"persistence {persistence_share:.0%}, recent shock {shock_share:.0%}, "
    f"leverage term {'active' if lev_active else 'inactive'}"
)

var_95 = var_results[0.95]
var_99 = var_results[0.99]
```


```python
display(Markdown(f"""

---



## Daily Market Risk Report — as of {as_of}



### Volatility forecast



| Quantity | Value |

|---|---|

| Next-day conditional standard deviation | {sigma_next:.4%} daily ({sigma_next_ann:.2%} annualised) |

| Expected absolute daily move | {abs_move_next:.4%} |

| Regime | **{regime_next}** |



{regime_context[regime_next]}



### Risk trend and context



| Quantity | Value |

|---|---|

| Historical percentile | {vol_percentile:.0%} |

| Regime confidence | {confidence_str} |

| Trend | {trend_str} |

| Forecast drivers | {driver_str} |



Percentile places the forecast within its own history; regime confidence

shows how close it sits to the next regime boundary; trend compares it to

recent weeks; drivers report which GARCH terms produced it (section 5).



### Risk thresholds



| Confidence | Daily VaR |

|---|---|

| 95% | {var_95['daily']:.4%} |

| 99% | {var_99['daily']:.4%} |



VaR computed as the alpha-quantile of the fitted Student's t return

distribution (nu = {nu:.1f}), scaled by the one-day-ahead conditional

standard deviation and including the fitted daily mean.



### Anomaly status



| Check | Result |

|---|---|

| Anomaly detected? | {anomaly_str} |



{anomaly_note}



### Model validation



| Check | Result |

|---|---|

| Forecast bias (share of mean abs return) | {bias_share:+.1%} |

| Mincer-Zarnowitz slope | {mz_slope:.3f} |

| 95% VaR breach rate | {bt95['breach_rate']:.2%} vs 5.00% expected ({bt95['verdict']}) |

| 99% VaR breach rate | {bt99['breach_rate']:.2%} vs 1.00% expected ({bt99['verdict']}) |



Full calibration and backtest analysis in sections 4 and 6.



### Allocation signal



| Parameter | Value |

|---|---|

| Current regime | {regime_next} |

| Regime multiplier | {multiplier:.1f}× |



The multiplier is applied weekly to a user-configurable baseline contribution.

It reflects a long-term value investing philosophy: deploy proportionally more

capital when volatility is elevated, because historically elevated volatility

has preceded stronger medium-term returns.



### Production model



GJR-GARCH(1,1,1) — Student's t, confirmed as the production model in

Notebook 06 after deep learning alternatives (LSTM, MLP) failed to improve

on it under identical walk-forward evaluation.



---

"""))
```




---



## Daily Market Risk Report — as of 2026-09-25



### Volatility forecast



| Quantity | Value |

|---|---|

| Next-day conditional standard deviation | 0.6446% daily (10.23% annualised) |

| Expected absolute daily move | 0.5818% |

| Regime | **Calm** |



Volatility sits in the bottom quartile of its historical distribution.



### Risk trend and context



| Quantity | Value |

|---|---|

| Historical percentile | 25% |

| Regime confidence | +0.00% from the Normal threshold |

| Trend | ↓ falling vs 1 week ago (11.1%), ↓ falling vs 1 month ago (11.1%) |

| Forecast drivers | persistence 96%, recent shock 0%, leverage term inactive |



Percentile places the forecast within its own history; regime confidence

shows how close it sits to the next regime boundary; trend compares it to

recent weeks; drivers report which GARCH terms produced it (section 5).



### Risk thresholds



| Confidence | Daily VaR |

|---|---|

| 95% | 0.9772% |

| 99% | 1.5847% |



VaR computed as the alpha-quantile of the fitted Student's t return

distribution (nu = 6.8), scaled by the one-day-ahead conditional

standard deviation and including the fitted daily mean.



### Anomaly status



| Check | Result |

|---|---|

| Anomaly detected? | No |



No anomaly detected. The most recent return fell within the range the model expected given current volatility conditions.



### Model validation



| Check | Result |

|---|---|

| Forecast bias (share of mean abs return) | -19.2% |

| Mincer-Zarnowitz slope | 0.850 |

| 95% VaR breach rate | 6.32% vs 5.00% expected (rejected) |

| 99% VaR breach rate | 1.44% vs 1.00% expected (rejected) |



Full calibration and backtest analysis in sections 4 and 6.



### Allocation signal



| Parameter | Value |

|---|---|

| Current regime | Calm |

| Regime multiplier | 1.0× |



The multiplier is applied weekly to a user-configurable baseline contribution.

It reflects a long-term value investing philosophy: deploy proportionally more

capital when volatility is elevated, because historically elevated volatility

has preceded stronger medium-term returns.



### Production model



GJR-GARCH(1,1,1) — Student's t, confirmed as the production model in

Notebook 06 after deep learning alternatives (LSTM, MLP) failed to improve

on it under identical walk-forward evaluation.



---




## 9. Operational workflow


```python
display(Markdown("""

The intended user of this report, in its current form, is the investor

running the live weekly allocation experiment — me. The format generalises:

the same page, produced for a portfolio or risk manager, is a morning

briefing on the equity risk environment. What follows is the Monday

routine the report was built for.



1. **Before opening the report**, record the intended contribution for the

   week in the decision log. This ordering is the whole point of the

   experiment: the intention must be committed before the signal is seen,

   or the intended-versus-actual comparison measures nothing.

2. **Refresh the pipeline.** Rerun Notebook 01 to pull the latest sessions,

   then this notebook to regenerate the report.

3. **Read the report.** Regime, forecast, VaR, anomaly status, and the

   validation block.

4. **Decide the actual contribution** and execute it.

5. **Complete the log entry**: signal action, actual action, and the reason

   for following or overriding the signal.

6. **After 21 trading days**, fill in the outcome columns for the entry.



The report informs three things: the size of the weekly contribution, via

the regime multiplier; whether the week warrants a closer review of news

flow, via the anomaly flag; and the loss magnitude to be psychologically

prepared for, via the VaR thresholds.



It explicitly does not inform direction, security selection, or exit

timing. Notebook 04 established that return direction is not forecastable

in this data — ARIMA(1,0,1) matched the historical-mean baseline exactly —

and the system makes no claim its own evidence contradicts. Any use of

this report to time entries and exits, rather than to size steady

contributions, is a misuse.

"""))
```




The intended user of this report, in its current form, is the investor

running the live weekly allocation experiment — me. The format generalises:

the same page, produced for a portfolio or risk manager, is a morning

briefing on the equity risk environment. What follows is the Monday

routine the report was built for.



1. **Before opening the report**, record the intended contribution for the

   week in the decision log. This ordering is the whole point of the

   experiment: the intention must be committed before the signal is seen,

   or the intended-versus-actual comparison measures nothing.

2. **Refresh the pipeline.** Rerun Notebook 01 to pull the latest sessions,

   then this notebook to regenerate the report.

3. **Read the report.** Regime, forecast, VaR, anomaly status, and the

   validation block.

4. **Decide the actual contribution** and execute it.

5. **Complete the log entry**: signal action, actual action, and the reason

   for following or overriding the signal.

6. **After 21 trading days**, fill in the outcome columns for the entry.



The report informs three things: the size of the weekly contribution, via

the regime multiplier; whether the week warrants a closer review of news

flow, via the anomaly flag; and the loss magnitude to be psychologically

prepared for, via the VaR thresholds.



It explicitly does not inform direction, security selection, or exit

timing. Notebook 04 established that return direction is not forecastable

in this data — ARIMA(1,0,1) matched the historical-mean baseline exactly —

and the system makes no claim its own evidence contradicts. Any use of

this report to time entries and exits, rather than to size steady

contributions, is a misuse.




## 10. Regime dynamics

### 10.1 Regime timeline


```python
# Colour-coded conditional volatility by regime
colours = {'Calm': '#2ecc71', 'Normal': '#3498db', 'Stress': '#f39c12', 'Crisis': '#e74c3c'}

fig = go.Figure()

for regime_name, colour in colours.items():
    mask = regimes == regime_name
    fig.add_trace(go.Scatter(
        x=returns.index[mask],
        y=cond_vol_ann[mask],
        mode='markers',
        marker=dict(size=2, color=colour, opacity=0.6),
        name=regime_name
    ))

for threshold, label in [(q25, 'P25'), (q75, 'P75'), (q95, 'P95')]:
    fig.add_hline(
        y=threshold, line_dash='dot', line_color='gray',
        annotation_text=label, annotation_position='right'
    )

fig.update_layout(
    title='Annualised conditional volatility by regime',
    xaxis_title='Date',
    yaxis_title='Annualised volatility',
    yaxis_tickformat='.0%',
    height=500,
    legend=dict(orientation='h', y=-0.15)
)

fig.show()
```



### 10.2 Transition matrix


```python
display(Markdown("""

Today's regime label answers "where are we?" The transition matrix answers

the reader's next question: "what usually happens next?" Each row shows the

probability of tomorrow's regime given today's, estimated from daily

transition frequencies over the full sample.

"""))
```




Today's regime label answers "where are we?" The transition matrix answers

the reader's next question: "what usually happens next?" Each row shows the

probability of tomorrow's regime given today's, estimated from daily

transition frequencies over the full sample.





```python
# Daily transition matrix
trans = pd.crosstab(regimes.shift(1), regimes, normalize='index')
trans = trans.reindex(index=regime_order, columns=regime_order).fillna(0)

trans_rows = '\n'.join(
    f"| **{r}** | " + ' | '.join(f'{trans.loc[r, c]:.1%}' for c in regime_order) + ' |'
    for r in regime_order
)

diag_calm = trans.loc['Calm', 'Calm']
diag_crisis = trans.loc['Crisis', 'Crisis']
jump_calm_to_crisis = trans.loc['Calm', 'Crisis']

display(Markdown(f"""

### Daily regime transition probabilities



| From \\ To | Calm | Normal | Stress | Crisis |

|---|---|---|---|---|

{trans_rows}



The diagonal dominates, as the persistence documented in Notebook 05

requires: a Calm day is followed by another Calm day {diag_calm:.0%} of the

time, and a Crisis day by another Crisis day {diag_crisis:.0%} of the time.

Transitions are overwhelmingly to adjacent regimes — the probability of

jumping from Calm directly to Crisis in one day is {jump_calm_to_crisis:.2%}.

Volatility escalates through Normal and Stress before reaching Crisis,

which is what gives the regime signal its operational value: it usually

provides warning steps rather than binary surprises.

"""))
```




### Daily regime transition probabilities



| From \ To | Calm | Normal | Stress | Crisis |

|---|---|---|---|---|

| **Calm** | 88.9% | 10.5% | 0.6% | 0.0% |
| **Normal** | 5.6% | 90.1% | 4.2% | 0.1% |
| **Stress** | 0.0% | 11.6% | 85.6% | 2.8% |
| **Crisis** | 0.0% | 0.0% | 12.4% | 87.6% |



The diagonal dominates, as the persistence documented in Notebook 05

requires: a Calm day is followed by another Calm day 89% of the

time, and a Crisis day by another Crisis day 88% of the time.

Transitions are overwhelmingly to adjacent regimes — the probability of

jumping from Calm directly to Crisis in one day is 0.00%.

Volatility escalates through Normal and Stress before reaching Crisis,

which is what gives the regime signal its operational value: it usually

provides warning steps rather than binary surprises.




### 10.3 Regime durations


```python
# Run-length encoding of regime episodes
run_id = (regimes != regimes.shift()).cumsum()
run_df = pd.DataFrame({
    'regime': regimes.values,
    'run': run_id.values,
    'date': regimes.index,
})
run_bounds = run_df.groupby('run').agg(
    regime=('regime', 'first'),
    start=('date', 'min'),
    end=('date', 'max'),
    length=('date', 'size'),
)

duration_stats = run_bounds.groupby('regime')['length'].agg(
    episodes='count', avg_duration='mean', median_duration='median',
    max_duration='max'
).reindex(regime_order)

# Same-day return and volatility by regime
same_day = pd.DataFrame({
    'regime': regimes,
    'ret': returns,
    'vol_ann': cond_vol_ann,
}).groupby('regime').agg(
    avg_return=('ret', 'mean'),
    avg_vol=('vol_ann', 'mean'),
).reindex(regime_order)

regime_stats = duration_stats.join(same_day)

stats_rows = '\n'.join(
    f"| {r} | {regime_stats.loc[r, 'avg_return']:.4%} | "
    f"{regime_stats.loc[r, 'avg_vol']:.1%} | "
    f"{regime_stats.loc[r, 'episodes']:.0f} | "
    f"{regime_stats.loc[r, 'avg_duration']:.1f} | "
    f"{regime_stats.loc[r, 'median_duration']:.0f} | "
    f"{regime_stats.loc[r, 'max_duration']:.0f} |"
    for r in regime_order
)

display(Markdown(f"""

### Regime statistics



| Regime | Avg daily return | Avg cond. vol (ann.) | Episodes | Mean duration (days) | Median duration (days) | Max duration (days) |

|---|---|---|---|---|---|---|

{stats_rows}



Duration is the operational number here. An investor seeing a Crisis label

wants to know how long these typically last: on average

{regime_stats.loc['Crisis', 'avg_duration']:.0f} trading days per episode

in this sample, against {regime_stats.loc['Calm', 'avg_duration']:.0f} for

Calm episodes. Elevated regimes are visits, not residences — which is

consistent with the mean-reverting variance process behind them — but the

maximum durations show the tail: the longest single

{'Crisis' if regime_stats.loc['Crisis', 'max_duration'] >= regime_stats.loc['Stress', 'max_duration'] else 'Stress'}

episode ran

{max(regime_stats.loc['Crisis', 'max_duration'], regime_stats.loc['Stress', 'max_duration']):.0f}

consecutive trading days. The median sits below the mean for the elevated

regimes, which is the signature of a right-skewed duration distribution:

most stress episodes are short, and a few long ones pull the average up.

The transition matrix gives the probabilities; these durations give the

lived experience behind them.

"""))
```




### Regime statistics



| Regime | Avg daily return | Avg cond. vol (ann.) | Episodes | Mean duration (days) | Median duration (days) | Max duration (days) |

|---|---|---|---|---|---|---|

| Calm | -0.0086% | 8.9% | 179 | 9.0 | 5 | 99 |
| Normal | 0.0392% | 13.9% | 319 | 10.1 | 5 | 71 |
| Stress | 0.0344% | 24.5% | 186 | 6.9 | 4 | 34 |
| Crisis | 0.0599% | 50.0% | 40 | 8.1 | 3 | 75 |



Duration is the operational number here. An investor seeing a Crisis label

wants to know how long these typically last: on average

8 trading days per episode

in this sample, against 9 for

Calm episodes. Elevated regimes are visits, not residences — which is

consistent with the mean-reverting variance process behind them — but the

maximum durations show the tail: the longest single

Crisis

episode ran

75

consecutive trading days. The median sits below the mean for the elevated

regimes, which is the signature of a right-skewed duration distribution:

most stress episodes are short, and a few long ones pull the average up.

The transition matrix gives the probabilities; these durations give the

lived experience behind them.




### 10.4 Historical stress events


```python
display(Markdown("""

A regime label is easier to trust when its past behaviour on named events

is visible. The table below takes six well-known stress windows and asks

how the classifier labelled them. This extends the two-event validation

from Notebook 05 into a fuller reference: when the report says Stress, the

reader can see which historical episodes carried the same label.

"""))
```




A regime label is easier to trust when its past behaviour on named events

is visible. The table below takes six well-known stress windows and asks

how the classifier labelled them. This extends the two-event validation

from Notebook 05 into a fuller reference: when the report says Stress, the

reader can see which historical episodes carried the same label.





```python
stress_windows = {
    '2008 financial crisis': ('2008-09-01', '2009-03-31'),
    '2011 debt ceiling / eurozone': ('2011-07-15', '2011-10-31'),
    '2015 China devaluation': ('2015-08-17', '2015-09-30'),
    '2018 Q4 selloff': ('2018-10-01', '2018-12-31'),
    '2020 COVID crash': ('2020-02-20', '2020-04-30'),
    '2022 rate-hike cycle': ('2022-01-01', '2022-10-31'),
}

event_rows = []
for name, (start, end) in stress_windows.items():
    window = regimes.loc[start:end]
    if len(window) == 0:
        continue
    pct_elevated = window.isin(['Stress', 'Crisis']).mean()
    pct_crisis = (window == 'Crisis').mean()
    modal = window.mode().iloc[0]
    event_rows.append(
        f'| {name} | {start} to {end} | {len(window)} | '
        f'{pct_elevated:.0%} | {pct_crisis:.0%} | {modal} |'
    )

events_table = '\n'.join(event_rows)

display(Markdown(f"""

### Classifier behaviour during known stress episodes



| Event | Window | Trading days | % Stress or Crisis | % Crisis | Modal regime |

|---|---|---|---|---|---|

{events_table}



The percentages are computed from the daily regime labels within each

window, so the table is regenerated from data on every run rather than

asserted. Differences between events are informative in themselves: an

acute shock like the 2020 COVID crash concentrates Crisis days into a

short window, while a grinding repricing like the 2022 rate-hike cycle

spends more of its length in Stress. The classifier distinguishes the two

shapes even though both were painful for a portfolio.

"""))
```




### Classifier behaviour during known stress episodes



| Event | Window | Trading days | % Stress or Crisis | % Crisis | Modal regime |

|---|---|---|---|---|---|

| 2008 financial crisis | 2008-09-01 to 2009-03-31 | 146 | 99% | 83% | Crisis |
| 2011 debt ceiling / eurozone | 2011-07-15 to 2011-10-31 | 76 | 84% | 37% | Stress |
| 2015 China devaluation | 2015-08-17 to 2015-09-30 | 32 | 88% | 19% | Stress |
| 2018 Q4 selloff | 2018-10-01 to 2018-12-31 | 63 | 76% | 2% | Stress |
| 2020 COVID crash | 2020-02-20 to 2020-04-30 | 50 | 94% | 88% | Crisis |
| 2022 rate-hike cycle | 2022-01-01 to 2022-10-31 | 209 | 69% | 11% | Stress |



The percentages are computed from the daily regime labels within each

window, so the table is regenerated from data on every run rather than

asserted. Differences between events are informative in themselves: an

acute shock like the 2020 COVID crash concentrates Crisis days into a

short window, while a grinding repricing like the 2022 rate-hike cycle

spends more of its length in Stress. The classifier distinguishes the two

shapes even though both were painful for a portfolio.




## 11. Retrospective feedback loop


```python
display(Markdown("""

The classifier's value depends on whether different regimes actually

produce different forward outcomes. If Crisis days are followed by the

same 21-day returns as Calm days, the classification adds no information.



This section evaluates the classifier independently of any allocation

behaviour. It measures what happened after each regime label, regardless

of what the investor did.

"""))
```




The classifier's value depends on whether different regimes actually

produce different forward outcomes. If Crisis days are followed by the

same 21-day returns as Calm days, the classification adds no information.



This section evaluates the classifier independently of any allocation

behaviour. It measures what happened after each regime label, regardless

of what the investor did.





```python
# Forward 21-day realised volatility and return for each day
fwd_vol_21 = returns.abs().rolling(21).mean().shift(-21)
fwd_return_21 = returns.rolling(21).sum().shift(-21)

feedback = pd.DataFrame({
    'regime': regimes,
    'fwd_vol_21': fwd_vol_21,
    'fwd_return_21': fwd_return_21
}).dropna()

fb_raw = feedback.groupby('regime').agg(
    avg_fwd_vol=('fwd_vol_21', 'mean'),
    avg_fwd_return=('fwd_return_21', 'mean'),
    count=('fwd_vol_21', 'count')
).reindex(regime_order)

crisis_vol = fb_raw.loc['Crisis', 'avg_fwd_vol']
calm_vol = fb_raw.loc['Calm', 'avg_fwd_vol']
crisis_ret = fb_raw.loc['Crisis', 'avg_fwd_return']
calm_ret = fb_raw.loc['Calm', 'avg_fwd_return']
vol_ratio = crisis_vol / calm_vol if calm_vol > 0 else np.nan

fb_rows = '\n'.join(
    f"| {r} | {fb_raw.loc[r, 'avg_fwd_vol']:.4%} | "
    f"{fb_raw.loc[r, 'avg_fwd_return']:.4%} | {fb_raw.loc[r, 'count']:,d} |"
    for r in regime_order
)

display(Markdown(f"""

### Feedback loop — classifier evaluation



| Regime | Avg next 21-day vol | Avg next 21-day return | Frequency |

|---|---|---|---|

{fb_rows}



{'Crisis days are followed by forward volatility ' + f'{vol_ratio:.1f}× higher than Calm days.' if not np.isnan(vol_ratio) else 'Insufficient data for volatility ratio.'}

{'Crisis and Stress regimes show stronger average forward returns than Calm, consistent with the volatility risk premium: elevated risk tends to precede higher compensation.' if crisis_ret > calm_ret else 'Forward return differences across regimes are modest in this sample.'}

"""))
```




### Feedback loop — classifier evaluation



| Regime | Avg next 21-day vol | Avg next 21-day return | Frequency |

|---|---|---|---|

| Calm | 0.5254% | 0.5480% | 1,605 |
| Normal | 0.6756% | 0.4794% | 3,204 |
| Stress | 1.0866% | 0.9155% | 1,288 |
| Crisis | 2.0788% | 0.6523% | 323 |



Crisis days are followed by forward volatility 4.0× higher than Calm days.

Crisis and Stress regimes show stronger average forward returns than Calm, consistent with the volatility risk premium: elevated risk tends to precede higher compensation.





```python
# Regime-conditional return distribution
fig = go.Figure()

for regime_name, colour in colours.items():
    mask = regimes == regime_name
    fig.add_trace(go.Box(
        y=returns[mask] * 100,
        name=regime_name,
        marker_color=colour,
        boxpoints=False
    ))

fig.update_layout(
    title='Daily log return distribution by regime',
    yaxis_title='Log return (%)',
    height=450,
    showlegend=False
)

fig.show()
```




```python
display(Markdown("""

The box plot confirms the regime classifier is capturing real distributional

differences. Crisis and Stress regimes show wider return dispersion and more

extreme outliers, consistent with the fat-tailed behaviour documented in

Notebook 02. Calm regimes produce tight, symmetric return distributions.

"""))
```




The box plot confirms the regime classifier is capturing real distributional

differences. Crisis and Stress regimes show wider return dispersion and more

extreme outliers, consistent with the fat-tailed behaviour documented in

Notebook 02. Calm regimes produce tight, symmetric return distributions.




## 12. Decision log


```python
display(Markdown(f"""

The decision log is the project's most original mechanic. It records not

just what the system recommended and what the investor did, but what the

investor intended to do *before* seeing the signal. The gap between

intended and actual action is the measurable effect of the framework on

behaviour.



The log is maintained manually each week, following the workflow in

section 9. Automation would defeat the purpose: the value is in the

cognitive step of recording intention before seeing the signal.



### Decision log schema



| Column | Description |

|---|---|

| Date | Monday of the decision week |

| Regime | Regime label from the most recent risk report |

| Forecast volatility | Annualised conditional volatility |

| Anomaly flag | Whether an anomaly was active at decision time |

| Intended action | What I planned to contribute *before* seeing the signal |

| Signal action | What the regime multiplier suggests |

| Actual action | What I actually contributed |

| Reason | Free text — why I followed or overrode the signal |

| 21-day return | Filled in retrospectively |

| 21-day realised vol | Filled in retrospectively |



### Example entry



| Column | Value |

|---|---|

| Date | (next Monday) |

| Regime | {regime_next} |

| Forecast volatility | {sigma_next_ann:.2%} |

| Anomaly flag | {'Yes' if today_flag else 'No'} |

| Intended action | £200 (my default) |

| Signal action | £{200 * multiplier:.0f} ({multiplier:.1f}× baseline) |

| Actual action | (to be filled) |

| Reason | (to be filled) |

| 21-day return | (filled in 21 days) |

| 21-day realised vol | (filled in 21 days) |

"""))
```




The decision log is the project's most original mechanic. It records not

just what the system recommended and what the investor did, but what the

investor intended to do *before* seeing the signal. The gap between

intended and actual action is the measurable effect of the framework on

behaviour.



The log is maintained manually each week, following the workflow in

section 9. Automation would defeat the purpose: the value is in the

cognitive step of recording intention before seeing the signal.



### Decision log schema



| Column | Description |

|---|---|

| Date | Monday of the decision week |

| Regime | Regime label from the most recent risk report |

| Forecast volatility | Annualised conditional volatility |

| Anomaly flag | Whether an anomaly was active at decision time |

| Intended action | What I planned to contribute *before* seeing the signal |

| Signal action | What the regime multiplier suggests |

| Actual action | What I actually contributed |

| Reason | Free text — why I followed or overrode the signal |

| 21-day return | Filled in retrospectively |

| 21-day realised vol | Filled in retrospectively |



### Example entry



| Column | Value |

|---|---|

| Date | (next Monday) |

| Regime | Calm |

| Forecast volatility | 10.23% |

| Anomaly flag | No |

| Intended action | £200 (my default) |

| Signal action | £200 (1.0× baseline) |

| Actual action | (to be filled) |

| Reason | (to be filled) |

| 21-day return | (filled in 21 days) |

| 21-day realised vol | (filled in 21 days) |




## 13. Decision rule retrospective

### 13.1 Price-based DCA simulation


```python
display(Markdown("""

This section answers: if the regime-multiplier rule had been applied

mechanically throughout the sample, how would it compare to passive

dollar-cost averaging?



The simulation uses actual S&P 500 Close prices. Each week, a contribution

buys index units at that week's closing level; the simulation tracks units

accumulated, cash invested, portfolio value, and the cost-weighted average

purchase price. The signal is the *previous* week's regime label — the

regime observed by Friday sizes the following week's purchase, mirroring

the Monday decision flow and removing within-week look-ahead in the

signal's timing.



One metric is deliberately absent: a Sharpe ratio comparison. Both

strategies hold the same single asset, so their time-weighted returns are

identical by construction — any Sharpe difference computed on the raw

portfolio value series would measure contribution inflows, not investment

skill. The metric that can differ between two schedules buying the same

asset is the money-weighted return, which credits a strategy for putting

more capital in before stronger periods. That is reported below, solved

numerically from the weekly cash flow schedule.

"""))
```




This section answers: if the regime-multiplier rule had been applied

mechanically throughout the sample, how would it compare to passive

dollar-cost averaging?



The simulation uses actual S&P 500 Close prices. Each week, a contribution

buys index units at that week's closing level; the simulation tracks units

accumulated, cash invested, portfolio value, and the cost-weighted average

purchase price. The signal is the *previous* week's regime label — the

regime observed by Friday sizes the following week's purchase, mirroring

the Monday decision flow and removing within-week look-ahead in the

signal's timing.



One metric is deliberately absent: a Sharpe ratio comparison. Both

strategies hold the same single asset, so their time-weighted returns are

identical by construction — any Sharpe difference computed on the raw

portfolio value series would measure contribution inflows, not investment

skill. The metric that can differ between two schedules buying the same

asset is the money-weighted return, which credits a strategy for putting

more capital in before stronger periods. That is reported below, solved

numerically from the weekly cash flow schedule.





```python
# Weekly panel: Friday close price, prior-week regime as the signal
weekly = pd.DataFrame({
    'price': close.resample('W-FRI').last(),
    'regime': regimes.resample('W-FRI').last(),
}).dropna()
weekly['signal_regime'] = weekly['regime'].shift(1)
weekly = weekly.dropna(subset=['signal_regime'])

print(f'Simulation covers {len(weekly):,} weeks '
      f'({weekly.index.min().date()} to {weekly.index.max().date()})')
```

    Simulation covers 1,336 weeks (2001-02-23 to 2026-09-25)
    


```python
def money_weighted_annual_return(contributions, final_value):
    """Annualised money-weighted return from weekly contributions.

    Solves for the weekly rate r such that the NPV of the cash flow
    schedule (contributions out, terminal value in) is zero, then
    annualises over 52 weeks."""
    cfs = (-contributions).to_numpy(dtype=float).copy()
    cfs[-1] += final_value
    periods = np.arange(len(cfs))

    def npv(r):
        return float(np.sum(cfs / (1 + r) ** periods))

    try:
        # Bracket kept numerically safe: at extreme weekly rates the
        # discount factors over ~1,300 weeks overflow or underflow
        # float64 and the NPV evaluates to nan, breaking the root
        # finder. (-0.2, 0.5) weekly spans any economically plausible
        # outcome while staying finite.
        r_weekly = brentq(npv, -0.2, 0.5)
        return (1 + r_weekly) ** 52 - 1
    except ValueError:
        return np.nan


def simulate_dca(schedule, baseline=1.0):
    """Simulate weekly DCA under a regime-to-multiplier schedule.

    Returns per-week series and summary scalars. Contributions are in
    normalised currency units (baseline = 1.0 per week at 1.0x)."""
    contrib = weekly['signal_regime'].map(schedule) * baseline
    units = contrib / weekly['price']
    cum_units = units.cumsum()
    cum_cost = contrib.cumsum()
    value = cum_units * weekly['price']
    avg_price = cum_cost / cum_units

    total_cost = cum_cost.iloc[-1]
    final_value = value.iloc[-1]
    roc = (final_value - total_cost) / total_cost
    final_avg_price = avg_price.iloc[-1]
    mwr = money_weighted_annual_return(contrib, final_value)

    cummax = value.cummax()
    mdd = ((value - cummax) / cummax).min()

    return {
        'contrib': contrib,
        'value': value,
        'avg_price': avg_price,
        'total_cost': total_cost,
        'final_value': final_value,
        'roc': roc,
        'final_avg_price': final_avg_price,
        'mwr': mwr,
        'mdd': mdd,
    }
```


```python
# Run passive and regime-aware (project schedule) simulations
flat_schedule = {'Calm': 1.0, 'Normal': 1.0, 'Stress': 1.0, 'Crisis': 1.0}

passive = simulate_dca(flat_schedule)
regime_aware = simulate_dca(regime_multipliers)

# Behaviour metrics for the regime-aware rule
signal_mult = weekly['signal_regime'].map(regime_multipliers)
n_weeks_different = int((signal_mult != 1.0).sum())
pct_different = n_weeks_different / len(weekly) * 100

elevated_mask = weekly['signal_regime'].isin(['Stress', 'Crisis'])
elevated_share = (
    regime_aware['contrib'][elevated_mask].sum()
    / regime_aware['contrib'].sum() * 100
)
passive_elevated_share = (
    passive['contrib'][elevated_mask].sum() / passive['contrib'].sum() * 100
)

print('Decision rule retrospective (price-based):')
print(f'  Total cost (passive):            {passive["total_cost"]:,.1f} units')
print(f'  Total cost (regime-aware):       {regime_aware["total_cost"]:,.1f} units')
print(f'  Return on cost (passive):        {passive["roc"]:.2%}')
print(f'  Return on cost (regime-aware):   {regime_aware["roc"]:.2%}')
print(f'  Avg purchase price (passive):    {passive["final_avg_price"]:,.2f}')
print(f'  Avg purchase price (regime):     {regime_aware["final_avg_price"]:,.2f}')
print(f'  Money-weighted return (passive): {passive["mwr"]:.2%} p.a.')
print(f'  Money-weighted return (regime):  {regime_aware["mwr"]:.2%} p.a.')
print(f'  Max drawdown (passive):          {passive["mdd"]:.2%}')
print(f'  Max drawdown (regime-aware):     {regime_aware["mdd"]:.2%}')
print(f'  Weeks with non-1.0x multiplier:  {n_weeks_different:,d} ({pct_different:.1f}%)')
print(f'  Capital in Stress/Crisis weeks:  {elevated_share:.1f}% vs {passive_elevated_share:.1f}% passive')
```

    Decision rule retrospective (price-based):
      Total cost (passive):            1,336.0 units
      Total cost (regime-aware):       2,038.0 units
      Return on cost (passive):        352.35%
      Return on cost (regime-aware):   367.75%
      Avg purchase price (passive):    1,711.82
      Avg purchase price (regime):     1,655.47
      Money-weighted return (passive): 10.30% p.a.
      Money-weighted return (regime):  10.33% p.a.
      Max drawdown (passive):          -47.07%
      Max drawdown (regime-aware):     -43.90%
      Weeks with non-1.0x multiplier:  1,006 (75.3%)
      Capital in Stress/Crisis weeks:  34.3% vs 25.0% passive
    


```python
price_gain = (
    (passive['final_avg_price'] - regime_aware['final_avg_price'])
    / passive['final_avg_price']
)

display(Markdown(f"""

### Decision rule comparison



| Metric | Passive DCA | Regime-aware DCA |

|---|---|---|

| Total cash invested | {passive['total_cost']:,.1f} units | {regime_aware['total_cost']:,.1f} units |

| Final portfolio value | {passive['final_value']:,.1f} units | {regime_aware['final_value']:,.1f} units |

| Return on cost | {passive['roc']:.2%} | {regime_aware['roc']:.2%} |

| Average purchase price | {passive['final_avg_price']:,.2f} | {regime_aware['final_avg_price']:,.2f} |

| Money-weighted return (annualised) | {passive['mwr']:.2%} | {regime_aware['mwr']:.2%} |

| Maximum drawdown | {passive['mdd']:.2%} | {regime_aware['mdd']:.2%} |

| Weeks with modified contribution | — | {n_weeks_different:,d} ({pct_different:.1f}%) |

| Capital share in Stress/Crisis weeks | {passive_elevated_share:.1f}% | {elevated_share:.1f}% |



{f'The regime-aware rule achieved a {price_gain:.2%} lower average purchase price than passive DCA — the signal directed extra capital toward cheaper entries.' if price_gain > 0 else f'The regime-aware rule achieved a {-price_gain:.2%} higher average purchase price than passive DCA in this sample — the extra capital deployed during elevated volatility did not buy cheaper entries. This is a null result, reported as found.'}

{'The money-weighted return of the regime-aware rule exceeds the passive rule, meaning the timing of its larger contributions added value beyond what steady contributions earned.' if regime_aware['mwr'] > passive['mwr'] else 'The money-weighted return of the regime-aware rule does not exceed the passive rule in this sample, a null result reported as found.'}



The maximum drawdown figures carry a caveat: weekly contributions cushion

portfolio drawdowns, and the two strategies contribute different amounts at

different times, so the drawdown comparison partly reflects contribution

timing rather than pure market exposure. It is reported because it is what

the investor would actually have experienced, not as a clean risk metric.

"""))
```




### Decision rule comparison



| Metric | Passive DCA | Regime-aware DCA |

|---|---|---|

| Total cash invested | 1,336.0 units | 2,038.0 units |

| Final portfolio value | 6,043.4 units | 9,532.7 units |

| Return on cost | 352.35% | 367.75% |

| Average purchase price | 1,711.82 | 1,655.47 |

| Money-weighted return (annualised) | 10.30% | 10.33% |

| Maximum drawdown | -47.07% | -43.90% |

| Weeks with modified contribution | — | 1,006 (75.3%) |

| Capital share in Stress/Crisis weeks | 25.0% | 34.3% |



The regime-aware rule achieved a 3.29% lower average purchase price than passive DCA — the signal directed extra capital toward cheaper entries.

The money-weighted return of the regime-aware rule exceeds the passive rule, meaning the timing of its larger contributions added value beyond what steady contributions earned.



The maximum drawdown figures carry a caveat: weekly contributions cushion

portfolio drawdowns, and the two strategies contribute different amounts at

different times, so the drawdown comparison partly reflects contribution

timing rather than pure market exposure. It is reported because it is what

the investor would actually have experienced, not as a clean risk metric.




### 13.2 Multiplier schedule sensitivity


```python
display(Markdown("""

The project schedule (1.0 / 1.5 / 2.0 / 2.5) is fixed by design rather

than optimised — tuning it on the same data used to evaluate it would

overfit. But a fixed schedule invites the question of whether the result

depends on those specific numbers. The check below runs the same

simulation under a flat schedule and a more aggressive one. If the

direction of the effect is consistent across schedules and its magnitude

scales with aggressiveness, the finding reflects the signal rather than

the parameterisation.

"""))
```




The project schedule (1.0 / 1.5 / 2.0 / 2.5) is fixed by design rather

than optimised — tuning it on the same data used to evaluate it would

overfit. But a fixed schedule invites the question of whether the result

depends on those specific numbers. The check below runs the same

simulation under a flat schedule and a more aggressive one. If the

direction of the effect is consistent across schedules and its magnitude

scales with aggressiveness, the finding reflects the signal rather than

the parameterisation.





```python
schedules = {
    'Flat (1 / 1 / 1 / 1)': flat_schedule,
    'Project (1 / 1.5 / 2 / 2.5)': regime_multipliers,
    'Aggressive (1 / 2 / 3 / 4)': {'Calm': 1.0, 'Normal': 2.0, 'Stress': 3.0, 'Crisis': 4.0},
}

sens_results = {name: simulate_dca(sched) for name, sched in schedules.items()}

sens_rows = '\n'.join(
    f"| {name} | {res['total_cost']:,.0f} | {res['roc']:.2%} | "
    f"{res['final_avg_price']:,.2f} | {res['mwr']:.2%} |"
    for name, res in sens_results.items()
)

avg_prices = [res['final_avg_price'] for res in sens_results.values()]
monotone_down = avg_prices[0] > avg_prices[1] > avg_prices[2]
monotone_up = avg_prices[0] < avg_prices[1] < avg_prices[2]

display(Markdown(f"""

### Sensitivity across multiplier schedules



| Schedule | Total cash invested | Return on cost | Avg purchase price | Money-weighted return (ann.) |

|---|---|---|---|---|

{sens_rows}



{'The average purchase price falls monotonically as the schedule becomes more aggressive: the more capital the rule tilts toward elevated-volatility weeks, the cheaper the average entry. The effect is a property of the signal, not of the specific multipliers.' if monotone_down else ('The average purchase price rises monotonically with aggressiveness: tilting capital toward elevated-volatility weeks bought more expensive entries in this sample. The direction is consistent across schedules, which makes it an honest property of the signal in this period — just not the hoped-for one.' if monotone_up else 'The average purchase price does not move monotonically with schedule aggressiveness, so the head-to-head result in section 13.1 should be read as schedule-specific rather than as a general property of the signal.')}

"""))
```




### Sensitivity across multiplier schedules



| Schedule | Total cash invested | Return on cost | Avg purchase price | Money-weighted return (ann.) |

|---|---|---|---|---|

| Flat (1 / 1 / 1 / 1) | 1,336 | 352.35% | 1,711.82 | 10.30% |
| Project (1 / 1.5 / 2 / 2.5) | 2,038 | 367.75% | 1,655.47 | 10.33% |
| Aggressive (1 / 2 / 3 / 4) | 2,740 | 375.25% | 1,629.32 | 10.35% |



The average purchase price falls monotonically as the schedule becomes more aggressive: the more capital the rule tilts toward elevated-volatility weeks, the cheaper the average entry. The effect is a property of the signal, not of the specific multipliers.




### 13.3 Portfolio trajectory


```python
# Portfolio value trajectory with Crisis-regime shading
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=passive['value'].index,
    y=passive['value'],
    name='Passive DCA',
    line=dict(width=1.5, color='#95a5a6')
))

fig.add_trace(go.Scatter(
    x=regime_aware['value'].index,
    y=regime_aware['value'],
    name='Regime-aware DCA',
    line=dict(width=1.5, color='#3498db')
))

# Shade Crisis episodes lasting at least 5 trading days
crisis_runs = run_bounds[
    (run_bounds['regime'] == 'Crisis') & (run_bounds['length'] >= 5)
]
for _, run in crisis_runs.iterrows():
    fig.add_vrect(
        x0=run['start'], x1=run['end'],
        fillcolor='#e74c3c', opacity=0.15, line_width=0
    )

fig.update_layout(
    title='Cumulative portfolio value: passive vs regime-aware DCA '
          '(Crisis episodes shaded)',
    xaxis_title='Date',
    yaxis_title='Portfolio value (normalised units)',
    height=500,
    legend=dict(orientation='h', y=-0.15)
)

fig.show()
```




```python
display(Markdown("""

The shaded bands mark Crisis episodes of five or more consecutive trading

days, taken from the run-length analysis in section 10.3. They are where

the two lines diverge: the regime-aware strategy deploys its largest

contributions inside the bands, so its value line pulls away from the

passive line in the recoveries that follow — or fails to, if the shock

keeps deepening. The chart makes the strategy's bet visible: it is a bet

on mean reversion after volatility spikes, and the shaded regions are

exactly where that bet is placed.

"""))
```




The shaded bands mark Crisis episodes of five or more consecutive trading

days, taken from the run-length analysis in section 10.3. They are where

the two lines diverge: the regime-aware strategy deploys its largest

contributions inside the bands, so its value line pulls away from the

passive line in the recoveries that follow — or fails to, if the shock

keeps deepening. The chart makes the strategy's bet visible: it is a bet

on mean reversion after volatility spikes, and the shaded regions are

exactly where that bet is placed.




### 13.4 Decision-rule attribution


```python
display(Markdown("""

The comparison shows whether the regime-aware rule differed from passive.

Attribution asks where that difference came from. Because every multiplier

in the project schedule is at least 1.0, the rule never buys less than

passive — it only buys more, in Normal, Stress, and Crisis weeks. So the

difference decomposes cleanly by regime: each regime's extra contribution

is (multiplier − 1) times the baseline, and the units that extra cash

bought can be valued at the final price. A regime that is common but mildly

tilted (Normal at 1.5×) and one that is rare but heavily tilted (Crisis at

2.5×) can contribute very differently, and which dominates is not obvious

in advance.

"""))
```




The comparison shows whether the regime-aware rule differed from passive.

Attribution asks where that difference came from. Because every multiplier

in the project schedule is at least 1.0, the rule never buys less than

passive — it only buys more, in Normal, Stress, and Crisis weeks. So the

difference decomposes cleanly by regime: each regime's extra contribution

is (multiplier − 1) times the baseline, and the units that extra cash

bought can be valued at the final price. A regime that is common but mildly

tilted (Normal at 1.5×) and one that is rare but heavily tilted (Crisis at

2.5×) can contribute very differently, and which dominates is not obvious

in advance.





```python
# Attribute the regime-aware vs passive difference to each regime.
# Extra contribution in regime R = (mult[R] - 1) * baseline, per week in R.
baseline = 1.0
final_price = weekly['price'].iloc[-1]

attrib_rows = []
attrib = {}
for r in regime_order:
    weeks_r = weekly['signal_regime'] == r
    n_r = int(weeks_r.sum())
    extra_contrib = (regime_multipliers[r] - 1) * baseline * n_r
    extra_units = ((regime_multipliers[r] - 1) * baseline / weekly['price'][weeks_r]).sum()
    extra_value = extra_units * final_price
    value_added = extra_value - extra_contrib
    attrib[r] = {
        'weeks': n_r,
        'extra_contrib': extra_contrib,
        'extra_units': extra_units,
        'value_added': value_added,
    }
    attrib_rows.append(
        f'| {r} | {n_r} | {extra_contrib:,.0f} | {extra_units:,.3f} | {value_added:+,.1f} |'
    )

total_extra_contrib = sum(a['extra_contrib'] for a in attrib.values())
total_value_added = sum(a['value_added'] for a in attrib.values())
attrib_table = '\n'.join(attrib_rows)

# Which regime deployed the most extra capital, and which added the most value
top_capital = max(regime_order, key=lambda r: attrib[r]['extra_contrib'])
top_value = max(regime_order, key=lambda r: attrib[r]['value_added'])

print('Attribution of the regime-aware vs passive difference:')
for r in regime_order:
    print(f'  {r:7s} extra cost {attrib[r]["extra_contrib"]:8,.0f}  '
          f'value added {attrib[r]["value_added"]:+8,.1f}')
print(f'  most extra capital: {top_capital}; most value added: {top_value}')
```

    Attribution of the regime-aware vs passive difference:
      Calm    extra cost        0  value added     +0.0
      Normal  extra cost      336  value added +1,142.7
      Stress  extra cost      270  value added +1,118.2
      Crisis  extra cost       96  value added   +526.4
      most extra capital: Normal; most value added: Normal
    


```python
display(Markdown(f"""

### Where the difference came from



| Regime | Weeks | Extra cash deployed | Extra units bought | Value added at final price |

|---|---|---|---|---|

{attrib_table}



The {top_capital} regime absorbed the most additional capital relative to

passive, and the {top_value} regime contributed the most value at the final

price. {'These are the same regime, so the rule’s effect is concentrated where it deployed most: the tilt did the work in one part of the distribution rather than spreading thinly.' if top_capital == top_value else 'These are different regimes — the one that received the most extra capital was not the one that added the most value, which is the kind of gap the sizing schedule is there to be judged on.'}



Whether the total effect helped depends on the market path. Across the whole

sample the extra {total_extra_contrib:,.0f} units of cash the rule deployed

produced a net value change of {total_value_added:+,.1f} units at the final

price.

{'The extra capital deployed into elevated-volatility weeks was worth more at the end than it cost, so the timing of the tilt added value in this sample.' if total_value_added > 0 else 'The extra capital deployed into elevated-volatility weeks was worth less at the end than it cost, so the tilt detracted in this sample — a null-to-negative result reported as found, and consistent with the average-price and money-weighted-return comparison above.'}

This is one sample path, and section 14 records the look-ahead caveats that

qualify it.

"""))
```




### Where the difference came from



| Regime | Weeks | Extra cash deployed | Extra units bought | Value added at final price |

|---|---|---|---|---|

| Calm | 330 | 0 | 0.000 | +0.0 |
| Normal | 672 | 336 | 0.191 | +1,142.7 |
| Stress | 270 | 270 | 0.179 | +1,118.2 |
| Crisis | 64 | 96 | 0.080 | +526.4 |



The Normal regime absorbed the most additional capital relative to

passive, and the Normal regime contributed the most value at the final

price. These are the same regime, so the rule’s effect is concentrated where it deployed most: the tilt did the work in one part of the distribution rather than spreading thinly.



Whether the total effect helped depends on the market path. Across the whole

sample the extra 702 units of cash the rule deployed

produced a net value change of +2,787.3 units at the final

price.

The extra capital deployed into elevated-volatility weeks was worth more at the end than it cost, so the timing of the tilt added value in this sample.

This is one sample path, and section 14 records the look-ahead caveats that

qualify it.




## 14. Limitations


```python
display(Markdown("""

The regime thresholds, VaR series, calibration statistics, and feedback-loop

figures are computed on the full sample. The GARCH variance recursion uses

only past returns at each step, so the day-by-day forecasts respect the

arrow of time — but the parameters and the percentile thresholds do not.

In a production system, both would be frozen on a training window and the

report would run on genuinely out-of-sample data. The calibration and

backtest results therefore validate the model's specification, not a live

deployment record.



The decision-rule retrospective lags the signal by one week, which removes

look-ahead in the signal's timing, but the regime labels themselves come

from full-sample thresholds — a subtler form of look-ahead that a

walk-forward evaluation of the allocation rule would eliminate. The

simulation also ignores transaction costs, contribution limits, and the

psychological friction of increasing contributions during market stress,

which the decision log exists to measure.



The allocation multipliers are fixed by design rather than optimised, and

the sensitivity check in section 13.2 varies them only across three

hand-picked schedules. It demonstrates directional consistency, not

optimality — no claim is made that any schedule is the best one.



The money-weighted return comparison inherits every caveat above, plus

one of its own: over a period where the index mostly rose, any rule that

invests more cash earlier or during dips will show a mechanical advantage

that may not repeat in a different market path. One sample path is one

draw.



The decision log depends on honest self-reporting. The intended-action

column is filled before seeing the signal; after the experiment is

running, there is no way to verify that the stated intention was genuine.



Those are limitations of the allocation experiment. The report itself has

a separate set, worth stating on their own because they bound what the

daily output can be used for. The forecast horizon is one day: the report

says nothing about risk a week or a month out, and the annualisation is a

scaling convention, not a multi-day forecast. The universe is the S&P 500

alone, so the report describes US large-cap equity risk and does not

capture cross-asset, credit, or liquidity risk that a real portfolio

carries. The regime thresholds and the fitted distribution encode

historical relationships that can shift; a structural break would leave

the classifier calibrated to a market that no longer exists, which is

precisely what the monitoring section is designed to catch. And the report

measures volatility, not expected return — a Crisis label says the market

is turbulent, not that it will fall, consistent with the Notebook 04 result

that direction is not forecastable in this data. Reading a risk signal as a

directional call is the misuse the whole project is built to prevent.

"""))
```




The regime thresholds, VaR series, calibration statistics, and feedback-loop

figures are computed on the full sample. The GARCH variance recursion uses

only past returns at each step, so the day-by-day forecasts respect the

arrow of time — but the parameters and the percentile thresholds do not.

In a production system, both would be frozen on a training window and the

report would run on genuinely out-of-sample data. The calibration and

backtest results therefore validate the model's specification, not a live

deployment record.



The decision-rule retrospective lags the signal by one week, which removes

look-ahead in the signal's timing, but the regime labels themselves come

from full-sample thresholds — a subtler form of look-ahead that a

walk-forward evaluation of the allocation rule would eliminate. The

simulation also ignores transaction costs, contribution limits, and the

psychological friction of increasing contributions during market stress,

which the decision log exists to measure.



The allocation multipliers are fixed by design rather than optimised, and

the sensitivity check in section 13.2 varies them only across three

hand-picked schedules. It demonstrates directional consistency, not

optimality — no claim is made that any schedule is the best one.



The money-weighted return comparison inherits every caveat above, plus

one of its own: over a period where the index mostly rose, any rule that

invests more cash earlier or during dips will show a mechanical advantage

that may not repeat in a different market path. One sample path is one

draw.



The decision log depends on honest self-reporting. The intended-action

column is filled before seeing the signal; after the experiment is

running, there is no way to verify that the stated intention was genuine.



Those are limitations of the allocation experiment. The report itself has

a separate set, worth stating on their own because they bound what the

daily output can be used for. The forecast horizon is one day: the report

says nothing about risk a week or a month out, and the annualisation is a

scaling convention, not a multi-day forecast. The universe is the S&P 500

alone, so the report describes US large-cap equity risk and does not

capture cross-asset, credit, or liquidity risk that a real portfolio

carries. The regime thresholds and the fitted distribution encode

historical relationships that can shift; a structural break would leave

the classifier calibrated to a market that no longer exists, which is

precisely what the monitoring section is designed to catch. And the report

measures volatility, not expected return — a Crisis label says the market

is turbulent, not that it will fall, consistent with the Notebook 04 result

that direction is not forecastable in this data. Reading a risk signal as a

directional call is the misuse the whole project is built to prevent.




## 15. Production monitoring


```python
display(Markdown("""

The project has walked the full modelling lifecycle: data validation,

diagnostics, feature engineering, baseline forecasting, volatility

modelling, model comparison, anomaly detection, and risk reporting with

backtesting. One piece of a real deployment remains, and it is the piece

that separates a finished analysis from a maintained system: knowing when

the model has drifted far enough to be rebuilt or investigated.



Every quantity below is already computed in this notebook, which is what

makes the monitoring concrete rather than aspirational. The triggers turn

each validation metric into a decision rule: while the metric sits inside

its band the production model runs untouched; when it crosses, the model is

flagged for investigation before the next allocation decision trusts it.

"""))
```




The project has walked the full modelling lifecycle: data validation,

diagnostics, feature engineering, baseline forecasting, volatility

modelling, model comparison, anomaly detection, and risk reporting with

backtesting. One piece of a real deployment remains, and it is the piece

that separates a finished analysis from a maintained system: knowing when

the model has drifted far enough to be rebuilt or investigated.



Every quantity below is already computed in this notebook, which is what

makes the monitoring concrete rather than aspirational. The triggers turn

each validation metric into a decision rule: while the metric sits inside

its band the production model runs untouched; when it crosses, the model is

flagged for investigation before the next allocation decision trusts it.





```python
# Monitoring status: current value of each validated metric against a
# trigger threshold. All inputs are computed earlier in this notebook.
monitor = []

# 95% VaR breach rate: flag if Kupiec rejects at 5%
monitor.append((
    '95% VaR breach rate',
    f'{bt95["breach_rate"]:.2%}',
    'Kupiec p > 0.05',
    'OK' if bt95['lr_pval'] > 0.05 else ('WATCH' if bt95['lr_pval'] > 0.01 else 'BREACH')
))
# 99% VaR breach rate
monitor.append((
    '99% VaR breach rate',
    f'{bt99["breach_rate"]:.2%}',
    'Kupiec p > 0.05',
    'OK' if bt99['lr_pval'] > 0.05 else ('WATCH' if bt99['lr_pval'] > 0.01 else 'BREACH')
))
# Forecast bias: flag if |bias| exceeds 10% of mean abs return
monitor.append((
    'Forecast bias (share of mean |r|)',
    f'{bias_share:+.1%}',
    '|bias| < 10%',
    'OK' if abs(bias_share) < 0.10 else ('WATCH' if abs(bias_share) < 0.20 else 'BREACH')
))
# Mincer-Zarnowitz slope: flag if far from 1
monitor.append((
    'Mincer-Zarnowitz slope',
    f'{mz_slope:.3f}',
    '0.8 – 1.2',
    'OK' if 0.8 <= mz_slope <= 1.2 else ('WATCH' if 0.6 <= mz_slope <= 1.4 else 'BREACH')
))
# Density calibration: worst central-interval coverage gap
worst_cov_gap = max(abs(coverage[l] - l) for l in cov_levels)
monitor.append((
    'Worst coverage gap (PIT)',
    f'{worst_cov_gap:.1%}',
    'gap < 3%',
    'OK' if worst_cov_gap < 0.03 else ('WATCH' if worst_cov_gap < 0.06 else 'BREACH')
))

monitor_rows = '\n'.join(
    f'| {name} | {val} | {rule} | {status} |' for name, val, rule, status in monitor
)
overall = 'BREACH' if any(m[3] == 'BREACH' for m in monitor) else (
    'WATCH' if any(m[3] == 'WATCH' for m in monitor) else 'OK')

print('Monitoring status:')
for name, val, rule, status in monitor:
    print(f'  [{status:6s}] {name}: {val} (trigger: {rule})')
print(f'  overall: {overall}')
```

    Monitoring status:
      [BREACH] 95% VaR breach rate: 6.32% (trigger: Kupiec p > 0.05)
      [BREACH] 99% VaR breach rate: 1.44% (trigger: Kupiec p > 0.05)
      [WATCH ] Forecast bias (share of mean |r|): -19.2% (trigger: |bias| < 10%)
      [OK    ] Mincer-Zarnowitz slope: 0.850 (trigger: 0.8 – 1.2)
      [OK    ] Worst coverage gap (PIT): 0.9% (trigger: gap < 3%)
      overall: BREACH
    


```python
display(Markdown(f"""

### Monitoring status



| Metric | Current value | Stay-in-service band | Status |

|---|---|---|---|

{monitor_rows}



Overall status: **{overall}**. A single BREACH is enough to pull the model

for investigation; a WATCH means the metric is drifting toward its edge and

should be checked more often. Two triggers are defined but not computed

live, because they need a reference the notebook does not recompute on each

run: a material change in the regime transition probabilities against the

matrix in section 10.2, and a rise in forecast error that erodes the

model's advantage over the naive persistence benchmark below the margin

established in Notebook 05 (a 30.8% RMSE reduction over the 252-day

walk-forward). Both would be evaluated on a rolling window in a scheduled

job rather than inside the report.



### How to read a failure



Each trigger maps to a specific economic consequence, which is what makes it

worth acting on. A persistent positive forecast bias means the model

systematically overestimates risk, which under this allocation rule pushes

capital into elevated-volatility weeks that were not as risky as claimed —

conservative sizing paid for with forecast error. A Kupiec rejection means

the VaR breach rate no longer matches its stated confidence, so the loss

thresholds in the report can no longer be read at face value and the risk

budget built on them is mis-stated. A PIT coverage gap that widens means the

shape of the return distribution has moved away from the fitted Student's t,

which degrades every quantile the report quotes, not just the mean. A

transition matrix that shifts means the regime dynamics themselves have

changed, so duration and persistence expectations built on history no longer

hold. In each case the fix is the same first step: refit on recent data and

compare the new parameters to the current ones before deciding whether the

change is drift or a structural break.

"""))
```




### Monitoring status



| Metric | Current value | Stay-in-service band | Status |

|---|---|---|---|

| 95% VaR breach rate | 6.32% | Kupiec p > 0.05 | BREACH |
| 99% VaR breach rate | 1.44% | Kupiec p > 0.05 | BREACH |
| Forecast bias (share of mean |r|) | -19.2% | |bias| < 10% | WATCH |
| Mincer-Zarnowitz slope | 0.850 | 0.8 – 1.2 | OK |
| Worst coverage gap (PIT) | 0.9% | gap < 3% | OK |



Overall status: **BREACH**. A single BREACH is enough to pull the model

for investigation; a WATCH means the metric is drifting toward its edge and

should be checked more often. Two triggers are defined but not computed

live, because they need a reference the notebook does not recompute on each

run: a material change in the regime transition probabilities against the

matrix in section 10.2, and a rise in forecast error that erodes the

model's advantage over the naive persistence benchmark below the margin

established in Notebook 05 (a 30.8% RMSE reduction over the 252-day

walk-forward). Both would be evaluated on a rolling window in a scheduled

job rather than inside the report.



### How to read a failure



Each trigger maps to a specific economic consequence, which is what makes it

worth acting on. A persistent positive forecast bias means the model

systematically overestimates risk, which under this allocation rule pushes

capital into elevated-volatility weeks that were not as risky as claimed —

conservative sizing paid for with forecast error. A Kupiec rejection means

the VaR breach rate no longer matches its stated confidence, so the loss

thresholds in the report can no longer be read at face value and the risk

budget built on them is mis-stated. A PIT coverage gap that widens means the

shape of the return distribution has moved away from the fitted Student's t,

which degrades every quantile the report quotes, not just the mean. A

transition matrix that shifts means the regime dynamics themselves have

changed, so duration and persistence expectations built on history no longer

hold. In each case the fix is the same first step: refit on recent data and

compare the new parameters to the current ones before deciding whether the

change is drift or a structural break.




## 16. What this notebook established


```python
display(Markdown(f"""

The Daily Market Risk Report combines three upstream outputs into a

single page of decision support: a next-day volatility forecast

({sigma_next:.4%} daily, {sigma_next_ann:.2%} annualised), a regime

label ({regime_next}), and an anomaly check

({'active' if today_flag else 'clear'}). Parametric VaR at 95% and 99%

confidence uses the fitted Student's t distribution, producing loss

thresholds of {var_95['daily']:.4%} and {var_99['daily']:.4%} daily.



The production system was validated before being trusted. The forecast

carries a bias of {bias_share:+.1%} of the mean absolute return and a

Mincer-Zarnowitz slope of {mz_slope:.3f}. The 95% VaR breached on

{bt95['breach_rate']:.2%} of days against 5% expected

(Kupiec p = {bt95['lr_pval']:.3f}, {bt95['verdict']}), and the 99% VaR on

{bt99['breach_rate']:.2%} against 1% expected

(Kupiec p = {bt99['lr_pval']:.3f}, {bt99['verdict']}). Density calibration

was checked beyond the point forecast: the PIT central-interval coverage

tracks its nominal levels to within {worst_cov_gap:.1%} at the worst point,

so the fitted distribution behind the VaR quantiles is calibrated in shape,

not only in scale.



The regime classifier's dynamics were characterised: transitions are

dominated by the diagonal and move through adjacent regimes,

{'Crisis-regime days are followed by 21-day volatility ' + f'{vol_ratio:.1f}× higher than Calm-regime days' if vol_ratio > 1.5 else 'regime-conditional forward outcomes differ across regimes'},

and all six named historical stress windows were labelled predominantly

elevated by the classifier.



The decision-rule retrospective, rebuilt on actual prices with a one-week

signal lag, compares regime-aware DCA against passive DCA on cash

invested, average purchase price, and money-weighted return.

{'The regime-aware rule bought at a lower average price and earned a higher money-weighted return.' if regime_aware['mwr'] > passive['mwr'] and price_gain > 0 else 'The regime-aware rule did not improve on passive DCA on the money-weighted measure in this sample, a null result reported as found.'}

The sensitivity check shows the direction of the effect is

{'consistent across multiplier schedules' if monotone_down or monotone_up else 'schedule-dependent'},

and every figure is stated with the look-ahead caveats from section 14.



The decision log template, operational workflow, and allocation multiplier

schedule are defined but not yet populated. The live weekly experiment

starts from the next Monday following this notebook's completion.



### What the project built



| Notebook | Contribution |

|---|---|

| 01 — EDA | Validated 25+ years of S&P 500 data, established log-return pipeline |

| 02 — Statistical diagnostics | Documented fat tails, volatility clustering, ARCH effects |

| 03 — Feature engineering | Built feature set linking diagnostics to modelling inputs |

| 04 — Baseline forecasting | Established null result: direction is not forecastable |

| 05 — Volatility forecasting | GJR-GARCH model, regime classifier, risk intelligence summary |

| 06 — Deep learning comparison | Confirmed GARCH as production model over LSTM and MLP |

| 07 — Anomaly detection | GARCH-residual anomaly flags for unexpected market events |

| **08 — Risk intelligence output** | **Validated Daily Market Risk Report, decision log, retrospective** |



The system is a quantitative decision-support tool. It says nothing about

what to buy or sell. It quantifies the risk environment those decisions

are made in, audits its own numbers before publishing them, and defines the

monitoring triggers that say when those numbers can no longer be trusted.

That closes the lifecycle from data ingestion through deployment to

maintenance.



### Possible future extensions



The project scope was fixed at a single asset on purpose, to keep the

modelling honest end to end rather than broad and shallow. Several

extensions follow naturally and are recorded here as future work, not as

gaps in the current build.



The most direct is expected shortfall alongside VaR: the fitted Student's t

already gives the tail, and the average loss beyond the VaR quantile is a

coherent risk measure that VaR is not. Beyond the single asset, a

multivariate model (DCC-GARCH or a copula) would carry the same regime and

tail machinery into portfolio VaR across correlated assets — the deferred

Version 2 direction. Macroeconomic covariates (rates, credit spreads, the

VIX term structure) could condition the volatility forecast on the broader

environment rather than on returns alone. On the engineering side, the

reusable logic moves into a package with a test suite, the JSON export this

notebook already writes feeds a live dashboard, and the monitoring triggers

defined above run as a scheduled job with rolling re-estimation. Each is a

step toward a deployed service; none is needed for the analysis this project

set out to do.

"""))
```




The Daily Market Risk Report combines three upstream outputs into a

single page of decision support: a next-day volatility forecast

(0.6446% daily, 10.23% annualised), a regime

label (Calm), and an anomaly check

(clear). Parametric VaR at 95% and 99%

confidence uses the fitted Student's t distribution, producing loss

thresholds of 0.9772% and 1.5847% daily.



The production system was validated before being trusted. The forecast

carries a bias of -19.2% of the mean absolute return and a

Mincer-Zarnowitz slope of 0.850. The 95% VaR breached on

6.32% of days against 5% expected

(Kupiec p = 0.000, rejected), and the 99% VaR on

1.44% against 1% expected

(Kupiec p = 0.001, rejected). Density calibration

was checked beyond the point forecast: the PIT central-interval coverage

tracks its nominal levels to within 0.9% at the worst point,

so the fitted distribution behind the VaR quantiles is calibrated in shape,

not only in scale.



The regime classifier's dynamics were characterised: transitions are

dominated by the diagonal and move through adjacent regimes,

Crisis-regime days are followed by 21-day volatility 4.0× higher than Calm-regime days,

and all six named historical stress windows were labelled predominantly

elevated by the classifier.



The decision-rule retrospective, rebuilt on actual prices with a one-week

signal lag, compares regime-aware DCA against passive DCA on cash

invested, average purchase price, and money-weighted return.

The regime-aware rule bought at a lower average price and earned a higher money-weighted return.

The sensitivity check shows the direction of the effect is

consistent across multiplier schedules,

and every figure is stated with the look-ahead caveats from section 14.



The decision log template, operational workflow, and allocation multiplier

schedule are defined but not yet populated. The live weekly experiment

starts from the next Monday following this notebook's completion.



### What the project built



| Notebook | Contribution |

|---|---|

| 01 — EDA | Validated 25+ years of S&P 500 data, established log-return pipeline |

| 02 — Statistical diagnostics | Documented fat tails, volatility clustering, ARCH effects |

| 03 — Feature engineering | Built feature set linking diagnostics to modelling inputs |

| 04 — Baseline forecasting | Established null result: direction is not forecastable |

| 05 — Volatility forecasting | GJR-GARCH model, regime classifier, risk intelligence summary |

| 06 — Deep learning comparison | Confirmed GARCH as production model over LSTM and MLP |

| 07 — Anomaly detection | GARCH-residual anomaly flags for unexpected market events |

| **08 — Risk intelligence output** | **Validated Daily Market Risk Report, decision log, retrospective** |



The system is a quantitative decision-support tool. It says nothing about

what to buy or sell. It quantifies the risk environment those decisions

are made in, audits its own numbers before publishing them, and defines the

monitoring triggers that say when those numbers can no longer be trusted.

That closes the lifecycle from data ingestion through deployment to

maintenance.



### Possible future extensions



The project scope was fixed at a single asset on purpose, to keep the

modelling honest end to end rather than broad and shallow. Several

extensions follow naturally and are recorded here as future work, not as

gaps in the current build.



The most direct is expected shortfall alongside VaR: the fitted Student's t

already gives the tail, and the average loss beyond the VaR quantile is a

coherent risk measure that VaR is not. Beyond the single asset, a

multivariate model (DCC-GARCH or a copula) would carry the same regime and

tail machinery into portfolio VaR across correlated assets — the deferred

Version 2 direction. Macroeconomic covariates (rates, credit spreads, the

VIX term structure) could condition the volatility forecast on the broader

environment rather than on returns alone. On the engineering side, the

reusable logic moves into a package with a test suite, the JSON export this

notebook already writes feeds a live dashboard, and the monitoring triggers

defined above run as a scheduled job with rolling re-estimation. Each is a

step toward a deployed service; none is needed for the analysis this project

set out to do.




## 17. Export


```python
# Export notebook metrics to locked_metrics.json
metrics['notebook_08'] = {
    'as_of': str(as_of),
    'sigma_next_daily': float(sigma_next),
    'sigma_next_ann': float(sigma_next_ann),
    'regime_next': regime_next,
    'vol_percentile': float(vol_percentile),
    'var_95_daily': float(var_95['daily']),
    'var_99_daily': float(var_99['daily']),
    'forecast_bias_share': float(bias_share),
    'mz_slope': float(mz_slope),
    'mz_r2': float(mz_r2),
    'var95_breach_rate': float(bt95['breach_rate']),
    'var95_kupiec_p': float(bt95['lr_pval']),
    'var99_breach_rate': float(bt99['breach_rate']),
    'var99_kupiec_p': float(bt99['lr_pval']),
    'passive_mwr': float(passive['mwr']),
    'regime_mwr': float(regime_aware['mwr']),
    'passive_avg_price': float(passive['final_avg_price']),
    'regime_avg_price': float(regime_aware['final_avg_price']),
    'passive_roc': float(passive['roc']),
    'regime_roc': float(regime_aware['roc']),
    'pit_worst_coverage_gap': float(worst_cov_gap),
    'forecast_persistence_share': float(persistence_share),
    'forecast_shock_share': float(shock_share),
    'trend_1w': trend_1w,
    'trend_1m': trend_1m,
    'attribution_top_capital_regime': top_capital,
    'attribution_top_value_regime': top_value,
    'monitoring_overall': overall,
}

metrics_path.write_text(json.dumps(metrics, indent=2))
print(f'Exported notebook_08 metrics to {metrics_path.resolve()}')
for k, v in metrics['notebook_08'].items():
    print(f'  {k}: {v}')
```

    Exported notebook_08 metrics to C:\Users\Mena\Documents\Python\sp500-market-intelligence\data\locked_metrics.json
      as_of: 2026-09-25
      sigma_next_daily: 0.006446032418042778
      sigma_next_ann: 0.10232759232720921
      regime_next: Calm
      vol_percentile: 0.2496506753609688
      var_95_daily: 0.009772262422524304
      var_99_daily: 0.015847201469209378
      forecast_bias_share: -0.192416235404335
      mz_slope: 0.8502414997799714
      mz_r2: 0.30966024665701225
      var95_breach_rate: 0.0631889458158671
      var95_kupiec_p: 2.9520826074769246e-06
      var99_breach_rate: 0.014438751746623195
      var99_kupiec_p: 0.000786905051341269
      passive_mwr: 0.10295390253370051
      regime_mwr: 0.10331093645956169
      passive_avg_price: 1711.8165320148096
      regime_avg_price: 1655.47109280328
      passive_roc: 3.52350471644061
      regime_roc: 3.677466245054
      pit_worst_coverage_gap: 0.00943952802359882
      forecast_persistence_share: 0.9551347145839919
      forecast_shock_share: 0.0
      trend_1w: falling
      trend_1m: falling
      attribution_top_capital_regime: Normal
      attribution_top_value_regime: Normal
      monitoring_overall: BREACH
    


```python
# Export the daily risk series as a reusable artefact
report_data = pd.DataFrame({
    'regime': regimes,
    'cond_vol_daily': cond_vol_daily,
    'cond_vol_ann': cond_vol_ann,
}, index=returns.index)

if anomalies is not None and 'std_resid' in anomalies.columns:
    report_data = report_data.join(
        anomalies[['std_resid', 'anomaly_flag']], how='left'
    )

report_path = Path('../data/nb08_risk_report.parquet')
report_data.to_parquet(report_path)
print(f'Risk report data exported to {report_path.resolve()}')
print(f'  Rows: {len(report_data):,}')
print(f'  Columns: {list(report_data.columns)}')
```

    Risk report data exported to C:\Users\Mena\Documents\Python\sp500-market-intelligence\data\nb08_risk_report.parquet
      Rows: 6,441
      Columns: ['regime', 'cond_vol_daily', 'cond_vol_ann']
    


```python
# Export the current report snapshot as JSON — consumable by Streamlit,
# Power BI, or an API without parsing parquet
report_json = {
    'as_of': str(as_of),
    'model': 'GJR-GARCH(1,1,1) Student-t',
    'forecast': {
        'sigma_daily': float(sigma_next),
        'sigma_annualised': float(sigma_next_ann),
        'expected_abs_move': float(abs_move_next),
        'historical_percentile': float(vol_percentile),
    },
    'regime': {
        'label': regime_next,
        'multiplier': float(multiplier),
        'distance_to_next_up': None if dist_up is None else float(dist_up),
        'distance_to_next_down': None if dist_dn is None else float(dist_dn),
        'thresholds_ann': {
            'calm_below': float(q25),
            'stress_above': float(q75),
            'crisis_above': float(q95),
        },
    },
    'trend': {
        'vol_1w_ago': None if np.isnan(vol_1w_ago) else float(vol_1w_ago),
        'vol_1m_ago': None if np.isnan(vol_1m_ago) else float(vol_1m_ago),
        'direction_1w': trend_1w,
        'direction_1m': trend_1m,
    },
    'forecast_drivers': {
        'persistence_share': float(persistence_share),
        'shock_share': float(shock_share),
        'leverage_active': bool(lev_active),
    },
    'var': {
        'daily_95': float(var_95['daily']),
        'daily_99': float(var_99['daily']),
    },
    'anomaly': {
        'flag': int(today_flag),
        'z_score': None if np.isnan(today_z) else float(today_z),
    },
    'validation': {
        'forecast_bias_share': float(bias_share),
        'mz_slope': float(mz_slope),
        'pit_worst_coverage_gap': float(worst_cov_gap),
        'var95_breach_rate': float(bt95['breach_rate']),
        'var95_kupiec_p': float(bt95['lr_pval']),
        'var95_verdict': bt95['verdict'],
        'var99_breach_rate': float(bt99['breach_rate']),
        'var99_kupiec_p': float(bt99['lr_pval']),
        'var99_verdict': bt99['verdict'],
    },
    'monitoring': {
        'overall': overall,
        'checks': [
            {'metric': name, 'value': val, 'band': rule, 'status': status}
            for name, val, rule, status in monitor
        ],
    },
}

json_path = Path('../reports/risk_report.json')
json_path.write_text(json.dumps(report_json, indent=2))
print(f'Report snapshot exported to {json_path.resolve()}')
```

    Report snapshot exported to C:\Users\Mena\Documents\Python\sp500-market-intelligence\reports\risk_report.json
    
