# 05 — Volatility forecasting

ARCH and GARCH models for one-day-ahead volatility forecasts, regime classification, and risk signal generation.

The notebook tells one continuous story: evidence that variance has structure, a benchmark that structure must beat, models of increasing realism, an out-of-sample test, and a decision-support output built on the winner.

## 1. Setup and objective


```python
# Import the needed libraries
import pandas as pd
import numpy as np
import plotly.io as pio
import warnings

from arch import arch_model
from arch.univariate import SkewStudent
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error
from IPython.display import display, Markdown
import json
from pathlib import Path
from scipy import stats
from scipy.special import gamma as gamma_fn
from scipy.stats import t as t_dist
from scipy.stats import binomtest

# Narrow the warnings filter: the blanket filterwarnings('ignore') used
# before hid genuine fit diagnostics, including boundary warnings the arch
# library raises when a coefficient pins at its constraint edge (see the
# GJR fit in section 5.5). Only cosmetic categories are suppressed here.
warnings.simplefilter('default')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Setting Plotly backend plotting for pandas
pd.options.plotting.backend = 'plotly'
pio.templates.default = 'plotly_dark'
```


```python
# Load locked figures established in earlier notebooks:

metrics = json.loads(Path('../data/locked_metrics.json').read_text())
nb02 = metrics['notebook_02']
nb04 = metrics['notebook_04']

skew_nb02 = nb02['skewness']
kurt_nb02 = nb02['excess_kurtosis']
jb_stat_nb02 = nb02['jarque_bera_stat']
jb_p_nb02 = nb02['jarque_bera_p']
arima_rmse_nb04 = nb04['arima_rmse']
hist_mean_rmse_nb04 = nb04['historical_mean_rmse']
rmse_tie = abs(arima_rmse_nb04 - hist_mean_rmse_nb04) < 1e-5

print(f"Notebook 02 — skewness: {skew_nb02:.4f}, excess kurtosis: {kurt_nb02:.4f}, "
      f"Jarque-Bera: {jb_stat_nb02:,.2f}")
print(f"Notebook 04 — ARIMA RMSE: {arima_rmse_nb04:.5f}, "
      f"historical-mean RMSE: {hist_mean_rmse_nb04:.5f}")
```

    Notebook 02 — skewness: -0.3487, excess kurtosis: 10.6850, Jarque-Bera: 32,055.27
    Notebook 04 — ARIMA RMSE: 0.01062, historical-mean RMSE: 0.01062
    


```python
display(Markdown(f"""
### Objective

Notebook 04 established that the direction of daily S&P 500 log returns is
difficult to forecast. ARIMA(1,0,1)
{'matched' if rmse_tie else 'scored close to'} the historical-mean baseline
(RMSE {arima_rmse_nb04:.5f}
{'in both cases' if rmse_tie else f'vs {hist_mean_rmse_nb04:.5f} for the baseline'}),
a null result consistent with weak-form market efficiency. The residual
diagnostics revealed structure in the *variance*: fat tails (excess kurtosis
{kurt_nb02:.4f}), negative skewness ({skew_nb02:.4f}), and persistent
clustering in the ACF of squared residuals.

This notebook targets that variance structure directly. The question is
narrow and testable:

> Do ARCH and GARCH-family models improve one-day-ahead forecasts of daily
> market volatility, relative to a simple persistence benchmark?

Daily realised volatility is approximated by the absolute daily log return,
a widely used proxy for latent daily volatility when high-frequency
realised-volatility estimates are unavailable. It has one property that
matters for honest evaluation: each forecast maps to exactly one realised
observation, with no overlapping windows, and its one-day horizon matches the
natural output of a GARCH(1,1) one-step forecast. The proxy is noisy but
informative — any single day is a poor read of latent volatility, but across
a 252-day test window that noise averages out enough to rank models fairly.

The 21-day rolling volatility feature (`vol_roll_21`) from Notebook 03 is not
used as the forecasting target here. It is a smoothed, overlapping-window
measure better suited to regime classification downstream than to one-day
model evaluation.
"""))
```



### Objective

Notebook 04 established that the direction of daily S&P 500 log returns is
difficult to forecast. ARIMA(1,0,1)
matched the historical-mean baseline
(RMSE 0.01062
in both cases),
a null result consistent with weak-form market efficiency. The residual
diagnostics revealed structure in the *variance*: fat tails (excess kurtosis
10.6850), negative skewness (-0.3487), and persistent
clustering in the ACF of squared residuals.

This notebook targets that variance structure directly. The question is
narrow and testable:

> Do ARCH and GARCH-family models improve one-day-ahead forecasts of daily
> market volatility, relative to a simple persistence benchmark?

Daily realised volatility is approximated by the absolute daily log return,
a widely used proxy for latent daily volatility when high-frequency
realised-volatility estimates are unavailable. It has one property that
matters for honest evaluation: each forecast maps to exactly one realised
observation, with no overlapping windows, and its one-day horizon matches the
natural output of a GARCH(1,1) one-step forecast. The proxy is noisy but
informative — any single day is a poor read of latent volatility, but across
a 252-day test window that noise averages out enough to rank models fairly.

The 21-day rolling volatility feature (`vol_roll_21`) from Notebook 03 is not
used as the forecasting target here. It is a smoothed, overlapping-window
measure better suited to regime classification downstream than to one-day
model evaluation.



### Notebook roadmap

1. Prepare the realised-volatility target.
2. Establish a persistence benchmark.
3. Test for conditional heteroskedasticity.
4. Fit competing ARCH- and GARCH-family models, one assumption relaxed per step.
5. Compare specifications using information criteria and select a winner.
6. Evaluate the selected model out of sample against the benchmark.
7. Convert forecasts into volatility regimes and a daily risk summary.

## 2. Data preparation

The first task is to isolate the daily log-return series, because volatility
models operate on returns rather than prices: prices are non-stationary,
while returns come far closer to satisfying the assumptions ARCH-family
models require. The cells below load the feature set built in Notebook 03,
verify the load, and extract the target series.


```python
# Load data
df = pd.read_parquet('../data/sp500_features.parquet')
```


```python
df.shape
```




    (6441, 66)




```python
df.head()
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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>log_returns</th>
      <th>return_lag_1</th>
      <th>return_lag_2</th>
      <th>return_lag_3</th>
      <th>return_lag_5</th>
      <th>...</th>
      <th>month_8</th>
      <th>month_9</th>
      <th>month_10</th>
      <th>month_11</th>
      <th>month_12</th>
      <th>vol_above_avg</th>
      <th>vol_rank_30</th>
      <th>high_vol_regime</th>
      <th>vol_expansion</th>
      <th>vol_ratio_10_60</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001-02-13</th>
      <td>1330.310059</td>
      <td>1336.619995</td>
      <td>1317.510010</td>
      <td>1318.800049</td>
      <td>1075200000</td>
      <td>-0.008690</td>
      <td>0.011758</td>
      <td>-0.013425</td>
      <td>-0.006254</td>
      <td>-0.001515</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.440476</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.614745</td>
    </tr>
    <tr>
      <th>2001-02-14</th>
      <td>1318.800049</td>
      <td>1320.729980</td>
      <td>1304.719971</td>
      <td>1315.920044</td>
      <td>1150300000</td>
      <td>-0.002186</td>
      <td>-0.008690</td>
      <td>0.011758</td>
      <td>-0.013425</td>
      <td>-0.008444</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.329365</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.618165</td>
    </tr>
    <tr>
      <th>2001-02-15</th>
      <td>1315.920044</td>
      <td>1331.290039</td>
      <td>1315.920044</td>
      <td>1326.609985</td>
      <td>1153700000</td>
      <td>0.008091</td>
      <td>-0.002186</td>
      <td>-0.008690</td>
      <td>0.011758</td>
      <td>-0.006254</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.194444</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.639375</td>
    </tr>
    <tr>
      <th>2001-02-16</th>
      <td>1326.609985</td>
      <td>1326.609985</td>
      <td>1293.180054</td>
      <td>1301.530029</td>
      <td>1257200000</td>
      <td>-0.019086</td>
      <td>0.008091</td>
      <td>-0.002186</td>
      <td>-0.008690</td>
      <td>-0.013425</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.250000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.656296</td>
    </tr>
    <tr>
      <th>2001-02-20</th>
      <td>1301.530029</td>
      <td>1307.160034</td>
      <td>1278.439941</td>
      <td>1278.939941</td>
      <td>1112200000</td>
      <td>-0.017509</td>
      <td>-0.019086</td>
      <td>0.008091</td>
      <td>-0.002186</td>
      <td>0.011758</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.162698</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.686551</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 66 columns</p>
</div>




```python
# Extract the target series
returns = df['log_returns'].dropna().sort_index()

n_obs = len(returns)
mean_daily_return = returns.mean()
daily_vol = returns.std()
annualized_vol = daily_vol * np.sqrt(252)
vol_to_mean_ratio = daily_vol / mean_daily_return
n_years = (returns.index.max() - returns.index.min()).days / 365.25

print(f'Observations: {n_obs:,}')
print(f'Mean daily return: {mean_daily_return:.6f}')
print(f'Daily volatility (std): {daily_vol:.6f}')
print(f'Annualized volatility: {annualized_vol:.4f} ({annualized_vol*100:.2f}%)')
```

    Observations: 6,441
    Mean daily return: 0.000273
    Daily volatility (std): 0.012037
    Annualized volatility: 0.1911 (19.11%)
    


```python
display(Markdown(f"""
The dataset contains {n_obs:,} daily observations spanning more than
{n_years:.0f} years, including periods of extreme stress (2008, 2020) and
extended low-volatility regimes.

Daily volatility ({daily_vol*100:.2f}%) is roughly {vol_to_mean_ratio:.0f}
times the mean daily return ({mean_daily_return*100:.3f}%). This imbalance is
typical in equity markets: returns are noisy and close to zero on average,
while volatility is persistent and structured. That asymmetry is why this
notebook shifts focus from return forecasting to volatility modelling.
"""))
```



The dataset contains 6,441 daily observations spanning more than
26 years, including periods of extreme stress (2008, 2020) and
extended low-volatility regimes.

Daily volatility (1.20%) is roughly 44
times the mean daily return (0.027%). This imbalance is
typical in equity markets: returns are noisy and close to zero on average,
while volatility is persistent and structured. That asymmetry is why this
notebook shifts focus from return forecasting to volatility modelling.



## 3. Persistence benchmark

Before any model is fitted, the bar it must clear is established: a forecast
that simply carries today's realised volatility into tomorrow.


```python
# Realised volatility proxy: the absolute daily log return
realised_vol = returns.abs()

# ── Benchmark 1: Persistence ──
benchmark = pd.DataFrame({'realised_vol': realised_vol})
benchmark['persistence_forecast'] = benchmark['realised_vol'].shift(1)

# ── Benchmark 2: EWMA (RiskMetrics, lambda = 0.94) ──
# The industry-standard naive volatility forecast. Conditional variance is an
# exponentially weighted moving average of squared returns, with a fixed decay
# and nothing estimated. Its square root is a conditional standard deviation,
# sigma, not an expected absolute return. RiskMetrics assumes conditionally
# Normal returns, so sigma converts to an expected absolute return with the
# Normal factor E|Z| = sqrt(2/pi). Every sigma forecaster in this notebook is
# converted before it is scored against |r|.
ewma_lambda = 0.94
ewma_abs_factor = np.sqrt(2 / np.pi)
sq_returns = returns ** 2
ewma_var = sq_returns.ewm(alpha=(1 - ewma_lambda), adjust=False).mean()
benchmark['ewma_forecast'] = np.sqrt(ewma_var.shift(1)) * ewma_abs_factor

benchmark = benchmark.dropna()

rmse_persistence = root_mean_squared_error(
    benchmark['realised_vol'],
    benchmark['persistence_forecast'])
mae_persistence = mean_absolute_error(
    benchmark['realised_vol'],
    benchmark['persistence_forecast']
)

rmse_ewma = root_mean_squared_error(
    benchmark['realised_vol'],
    benchmark['ewma_forecast'])
mae_ewma = mean_absolute_error(
    benchmark['realised_vol'],
    benchmark['ewma_forecast']
)

print(f'Persistence Benchmark  RMSE: {rmse_persistence:.6f}  MAE: {mae_persistence:.6f}')
print(f'EWMA (λ={ewma_lambda})       RMSE: {rmse_ewma:.6f}  MAE: {mae_ewma:.6f}')
```

    Persistence Benchmark  RMSE: 0.010770  MAE: 0.007109
    EWMA (λ=0.94)       RMSE: 0.007874  MAE: 0.005364
    


```python
display(Markdown(f"""
Before fitting any conditional volatility model, two naive baselines set the
bar.

**Persistence** assumes tomorrow's realised volatility equals today's:
`persistence_forecast(t) = realised_vol(t-1)`. It is deliberately simple, and
it works because volatility clusters: today's level carries real information
about tomorrow's. Over the full sample it produced
RMSE = {rmse_persistence:.6f} and MAE = {mae_persistence:.6f}.

**EWMA (RiskMetrics, λ = {ewma_lambda})** is the industry-standard naive
forecast. Conditional variance is an exponentially weighted moving average of
squared returns, the same smooth-decay structure GARCH formalises, but with a
fixed decay parameter and nothing estimated. Its output is a sigma, so it is
converted to an expected absolute return with the Normal factor its
RiskMetrics design assumes (sqrt(2/pi) ≈ {ewma_abs_factor:.4f}), the same
units every other forecaster is scored in. Over the full sample it produced
RMSE = {rmse_ewma:.6f} and MAE = {mae_ewma:.6f}.

EWMA is the harder benchmark. Persistence copies a single noisy observation
forward; EWMA smooths the history, so its forecast is less volatile and
closer to latent volatility on average. Any model that beats persistence but
not EWMA has learned smoothing, not structure. Every model in this notebook
is evaluated against both baselines on an identical realised-volatility
target.
"""))
```



Before fitting any conditional volatility model, two naive baselines set the
bar.

**Persistence** assumes tomorrow's realised volatility equals today's:
`persistence_forecast(t) = realised_vol(t-1)`. It is deliberately simple, and
it works because volatility clusters: today's level carries real information
about tomorrow's. Over the full sample it produced
RMSE = 0.010770 and MAE = 0.007109.

**EWMA (RiskMetrics, λ = 0.94)** is the industry-standard naive
forecast. Conditional variance is an exponentially weighted moving average of
squared returns, the same smooth-decay structure GARCH formalises, but with a
fixed decay parameter and nothing estimated. Its output is a sigma, so it is
converted to an expected absolute return with the Normal factor its
RiskMetrics design assumes (sqrt(2/pi) ≈ 0.7979), the same
units every other forecaster is scored in. Over the full sample it produced
RMSE = 0.007874 and MAE = 0.005364.

EWMA is the harder benchmark. Persistence copies a single noisy observation
forward; EWMA smooths the history, so its forecast is less volatile and
closer to latent volatility on average. Any model that beats persistence but
not EWMA has learned smoothing, not structure. Every model in this notebook
is evaluated against both baselines on an identical realised-volatility
target.



## 4. Evidence of conditional heteroskedasticity

Modelling time-varying variance is only justified if variance actually
varies, with structure. The eyeball test comes first — the returns plot makes
the clustering visible — and Engle's ARCH-LM test then confirms it formally.


```python
returns.plot(
    title='S&P 500 Daily Log Returns',
    labels={'value': 'Log Returns', 'index': 'Date'}
)
```




```python
# Engle's ARCH-LM test
arch_test = het_arch(returns, nlags=20)

print(f'LM Statistic: {arch_test[0]:.2f}')
print(f'P-value: {arch_test[1]:.6f}')
```

    LM Statistic: 1867.15
    P-value: 0.000000
    


```python
display(Markdown(f"""
### Engle's ARCH-LM test

Before fitting GARCH-family models, the return series must exhibit
conditional heteroskedasticity — variance that changes over time rather than
remaining constant.

Engle's ARCH-LM test was applied to the daily log returns using 20 lags. The
test returned **LM = {arch_test[0]:,.2f}**
(**p {'≈ 0' if arch_test[1] < 0.0001 else f'= {arch_test[1]:.4f}'}**),
rejecting the null hypothesis of constant variance.

This is consistent with Notebook 02, where the ACF of squared returns showed
persistent volatility clustering. Periods of elevated volatility tend to be
followed by further elevated volatility, and calm periods persist similarly.

The variance process contains systematic structure that constant-variance
assumptions cannot capture. The persistence benchmark exploits that structure
by copying it forward; the models in the next section estimate it.
"""))
```



### Engle's ARCH-LM test

Before fitting GARCH-family models, the return series must exhibit
conditional heteroskedasticity — variance that changes over time rather than
remaining constant.

Engle's ARCH-LM test was applied to the daily log returns using 20 lags. The
test returned **LM = 1,867.15**
(**p ≈ 0**),
rejecting the null hypothesis of constant variance.

This is consistent with Notebook 02, where the ACF of squared returns showed
persistent volatility clustering. Periods of elevated volatility tend to be
followed by further elevated volatility, and calm periods persist similarly.

The variance process contains systematic structure that constant-variance
assumptions cannot capture. The persistence benchmark exploits that structure
by copying it forward; the models in the next section estimate it.



## 5. Model estimation

Model estimation asks which specification best explains the observed data.
Forecast evaluation (section 6) asks which specification predicts unseen
observations. The two questions get different answers often enough that this
notebook keeps them in separate sections with separate metrics:
log-likelihood, AIC, and BIC here; walk-forward RMSE and MAE there.

Five specifications are fitted in a controlled sequence, each changing one
thing: ARCH(1) tests the mechanism, GARCH(1,1) adds memory, Student's t fixes
the tails, GJR-GARCH tests asymmetry in how variance responds to shocks, and a
skewed Student's t tests asymmetry in the shocks themselves.

Each model registers itself in a
shared results table immediately after fitting, and the table is re-displayed
after every registration, so the standings are visible at each step rather
than revealed once at the end. The AIC winner advances to forecast
evaluation and powers the regime classifier and risk summary.

### 5.1 Fitting decisions


```python
display(Markdown(f"""
Three choices apply to every model in this notebook and are fixed here.

**Scale.** Returns are multiplied by 100 before fitting. Daily log returns
have variance of order 1e-4, small enough that the optimiser's default
convergence tolerances can fail. Fitting in percentage points is the standard
remedy. The choice is purely numerical: the statistical model is
unchanged. Log-likelihoods shift by a constant under rescaling, and that constant
is identical for every model fitted on the same series, so AIC and
log-likelihood comparisons between models are unaffected. The one obligation
this creates: every conditional volatility forecast is divided by 100 before
comparison against the realised-volatility target, which remains in raw
return units.

**Mean equation.** A constant mean. Notebook 04 established that the
conditional mean of daily returns carries no exploitable structure —
ARIMA(1,0,1) matched the historical-mean baseline exactly. Modelling the mean
beyond a constant would add parameters with no evidence behind them.

**Innovation distribution.** Normal, to start. This is a deliberate
simplification and a known misspecification: excess kurtosis of {kurt_nb02:.4f} and
skewness of {skew_nb02:.4f} were documented in Notebook 02. The Normal assumption is
kept for the first two fits so that, when Student's t innovations are
introduced later, the improvement is attributable to the distributional
change alone.
"""))
```



Three choices apply to every model in this notebook and are fixed here.

**Scale.** Returns are multiplied by 100 before fitting. Daily log returns
have variance of order 1e-4, small enough that the optimiser's default
convergence tolerances can fail. Fitting in percentage points is the standard
remedy. The choice is purely numerical: the statistical model is
unchanged. Log-likelihoods shift by a constant under rescaling, and that constant
is identical for every model fitted on the same series, so AIC and
log-likelihood comparisons between models are unaffected. The one obligation
this creates: every conditional volatility forecast is divided by 100 before
comparison against the realised-volatility target, which remains in raw
return units.

**Mean equation.** A constant mean. Notebook 04 established that the
conditional mean of daily returns carries no exploitable structure —
ARIMA(1,0,1) matched the historical-mean baseline exactly. Modelling the mean
beyond a constant would add parameters with no evidence behind them.

**Innovation distribution.** Normal, to start. This is a deliberate
simplification and a known misspecification: excess kurtosis of 10.6850 and
skewness of -0.3487 were documented in Notebook 02. The Normal assumption is
kept for the first two fits so that, when Student's t innovations are
introduced later, the improvement is attributable to the distributional
change alone.




```python
# Model results collector — single source of truth for the comparison table.
# Each model registers its stats:
model_results = {}
model_objects = {}
model_specs = {}

def register_model(label, result, spec):
    model_results[label] = {
        'log_likelihood': result.loglikelihood,
        'aic': result.aic,
        'bic': result.bic,
        'n_params': result.params.shape[0],
    }
    model_objects[label] = result
    model_specs[label] = dict(spec)

def show_model_table():
    table = pd.DataFrame(model_results).T
    table = table[['log_likelihood', 'aic', 'bic', 'n_params']].sort_values('aic')
    table['delta_aic'] = table['aic'] - table['aic'].min()
    table['delta_ll'] = table['log_likelihood'] - table['log_likelihood'].max()
    display(table)
```

### 5.2 ARCH(1)


```python
display(Markdown(f"""
The persistence benchmark copies volatility forward: tomorrow equals today. It
works because volatility is persistent, but it explains nothing. It has no
model of *how* variance evolves, only the assumption that it changes slowly.

ARCH (Autoregressive Conditional Heteroskedasticity) replaces that assumption
with an estimated process. Rather than holding variance constant or copying it
forward, it models today's conditional variance as a function of recent
shocks, the squared surprises in past returns. A large move yesterday raises
the variance expected today; a calm day lowers it. This is the mechanism
behind volatility clustering, and it is the structure the ARCH-LM test
(LM = {arch_test[0]:,.2f},
p {'< 0.001' if arch_test[1] < 0.001 else f'= {arch_test[1]:.4f}'})
detected.

The simplest specification is ARCH(1): conditional variance depends on a
constant and a single lag of the squared shock. It is intentionally minimal.
It tests one thing — whether yesterday's shock alone carries usable
information about today's variance — before any persistence structure is
added.
"""))
```



The persistence benchmark copies volatility forward: tomorrow equals today. It
works because volatility is persistent, but it explains nothing. It has no
model of *how* variance evolves, only the assumption that it changes slowly.

ARCH (Autoregressive Conditional Heteroskedasticity) replaces that assumption
with an estimated process. Rather than holding variance constant or copying it
forward, it models today's conditional variance as a function of recent
shocks, the squared surprises in past returns. A large move yesterday raises
the variance expected today; a calm day lowers it. This is the mechanism
behind volatility clustering, and it is the structure the ARCH-LM test
(LM = 1,867.15,
p < 0.001)
detected.

The simplest specification is ARCH(1): conditional variance depends on a
constant and a single lag of the squared shock. It is intentionally minimal.
It tests one thing — whether yesterday's shock alone carries usable
information about today's variance — before any persistence structure is
added.




```python
# Fit ARCH(1) on percentage returns
returns_scaled = returns * 100

spec_arch1 = dict(mean='Constant', vol='ARCH', p=1, dist='normal')
res_arch1 = arch_model(returns_scaled, **spec_arch1).fit(disp='off')

print(res_arch1.summary())
```

                          Constant Mean - ARCH Model Results                      
    ==============================================================================
    Dep. Variable:            log_returns   R-squared:                       0.000
    Mean Model:             Constant Mean   Adj. R-squared:                  0.000
    Vol Model:                       ARCH   Log-Likelihood:               -9812.15
    Distribution:                  Normal   AIC:                           19630.3
    Method:            Maximum Likelihood   BIC:                           19650.6
                                            No. Observations:                 6441
    Date:                Wed, Oct 07 2026   Df Residuals:                     6440
    Time:                        10:45:06   Df Model:                            1
                                     Mean Model                                 
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    mu             0.0577  1.575e-02      3.665  2.478e-04 [2.684e-02,8.856e-02]
                                Volatility Model                            
    ========================================================================
                     coef    std err          t      P>|t|  95.0% Conf. Int.
    ------------------------------------------------------------------------
    omega          0.9084  4.728e-02     19.211  3.001e-82 [  0.816,  1.001]
    alpha[1]       0.4013  5.486e-02      7.315  2.578e-13 [  0.294,  0.509]
    ========================================================================
    
    Covariance estimator: robust
    


```python
register_model('ARCH(1) — Normal', res_arch1, spec_arch1)
show_model_table()
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
      <th>log_likelihood</th>
      <th>aic</th>
      <th>bic</th>
      <th>n_params</th>
      <th>delta_aic</th>
      <th>delta_ll</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ARCH(1) — Normal</th>
      <td>-9812.146411</td>
      <td>19630.292822</td>
      <td>19650.60414</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Test whether one lag of squared shocks absorbed the clustering
std_resid_arch1 = res_arch1.std_resid.dropna()
arch_test_resid1 = het_arch(std_resid_arch1, nlags=20)
lb_sq_arch1 = acorr_ljungbox(std_resid_arch1 ** 2, lags=[20], return_df=True)
lb_stat_a1 = lb_sq_arch1['lb_stat'].iloc[0]
lb_p_a1 = lb_sq_arch1['lb_pvalue'].iloc[0]

omega_a1 = res_arch1.params['omega']
alpha_a1 = res_arch1.params['alpha[1]']
alpha_a1_p = res_arch1.pvalues['alpha[1]']

print(f'omega: {omega_a1:.4f}')
print(f'alpha[1]: {alpha_a1:.4f} (p = {alpha_a1_p:.2e})')
print(f'ARCH-LM on standardised residuals: LM = {arch_test_resid1[0]:.2f}, '
      f'p = {arch_test_resid1[1]:.6f}')
print(f'Ljung-Box on squared std residuals (lag 20): '
      f'Q = {lb_stat_a1:.2f}, p = {lb_p_a1:.6f}')

display(Markdown(f"""
#### Interpreting ARCH(1)

The single ARCH coefficient is alpha[1] = {alpha_a1:.4f}
(p {'< 0.001' if alpha_a1_p < 0.001 else f'= {alpha_a1_p:.4f}'}). Yesterday's
squared shock carries real information about today's variance, confirming at
the model level what the ARCH-LM test showed at the data level.

But the model's memory is exactly one day. Under ARCH(1), a large shock
raises expected variance for a single day and then vanishes from the
information set. The stylised fact from Notebook 04 was different: the ACF of
squared residuals showed clustering that persists across many lags, not one.

The standardised residuals make this failure measurable. If ARCH(1) had
captured the variance dynamics, its standardised residuals would show no
remaining ARCH effects. The test returns
LM = {arch_test_resid1[0]:,.2f}
(p {'< 0.001' if arch_test_resid1[1] < 0.001 else f'= {arch_test_resid1[1]:.4f}'}).
{'Significant conditional heteroskedasticity survives the fit. One lag of squared shocks is not enough.' if arch_test_resid1[1] < 0.05 else 'Remaining ARCH effects are not significant at the 5% level — an unexpected result worth verifying before proceeding.'}

The Ljung-Box test on squared standardised residuals corroborates
(Q = {lb_stat_a1:,.2f},
p {'< 0.001' if lb_p_a1 < 0.001 else f'= {lb_p_a1:.4f}'}):
{'serial dependence in the squared residuals persists.' if lb_p_a1 < 0.05 else 'no significant serial dependence remains — surprising, and inconsistent with the ARCH-LM result.'}
"""))
```

    omega: 0.9084
    alpha[1]: 0.4013 (p = 2.58e-13)
    ARCH-LM on standardised residuals: LM = 1089.33, p = 0.000000
    Ljung-Box on squared std residuals (lag 20): Q = 2470.41, p = 0.000000
    



#### Interpreting ARCH(1)

The single ARCH coefficient is alpha[1] = 0.4013
(p < 0.001). Yesterday's
squared shock carries real information about today's variance, confirming at
the model level what the ARCH-LM test showed at the data level.

But the model's memory is exactly one day. Under ARCH(1), a large shock
raises expected variance for a single day and then vanishes from the
information set. The stylised fact from Notebook 04 was different: the ACF of
squared residuals showed clustering that persists across many lags, not one.

The standardised residuals make this failure measurable. If ARCH(1) had
captured the variance dynamics, its standardised residuals would show no
remaining ARCH effects. The test returns
LM = 1,089.33
(p < 0.001).
Significant conditional heteroskedasticity survives the fit. One lag of squared shocks is not enough.

The Ljung-Box test on squared standardised residuals corroborates
(Q = 2,470.41,
p < 0.001):
serial dependence in the squared residuals persists.



### 5.3 GARCH(1,1), Normal innovations


```python
display(Markdown("""
ARCH(1) answered its question: yesterday's shock does move today's
variance. But the model can carry information only through the most recent
shock, while volatility in this data decays gradually rather than vanishing
after a day.

The direct fix for ARCH(1)'s one-day memory is more lags: ARCH(5), ARCH(10),
ARCH(20). Each added lag is another estimated parameter, and long-memory
volatility would demand many of them. That path trades one problem
(short memory) for another (parameter proliferation and unstable estimates).

GARCH(1,1) solves this with a single extra parameter. Alongside the lagged
squared shock (alpha), it adds the lagged conditional variance itself (beta).
Because yesterday's conditional variance already summarises the entire shock
history, this recursion is equivalent to an ARCH model with infinitely many
geometrically decaying lags — long memory at the cost of one coefficient.

The sum alpha + beta measures persistence: how slowly a volatility shock
decays. Values near 1 mean shocks take weeks or months to fade, which is
exactly the behaviour the ACF of squared residuals displayed. For equity
indices, estimates typically land between 0.95 and 1.00. Where this series
lands is the first thing to check in the fit below.
"""))
```



ARCH(1) answered its question: yesterday's shock does move today's
variance. But the model can carry information only through the most recent
shock, while volatility in this data decays gradually rather than vanishing
after a day.

The direct fix for ARCH(1)'s one-day memory is more lags: ARCH(5), ARCH(10),
ARCH(20). Each added lag is another estimated parameter, and long-memory
volatility would demand many of them. That path trades one problem
(short memory) for another (parameter proliferation and unstable estimates).

GARCH(1,1) solves this with a single extra parameter. Alongside the lagged
squared shock (alpha), it adds the lagged conditional variance itself (beta).
Because yesterday's conditional variance already summarises the entire shock
history, this recursion is equivalent to an ARCH model with infinitely many
geometrically decaying lags — long memory at the cost of one coefficient.

The sum alpha + beta measures persistence: how slowly a volatility shock
decays. Values near 1 mean shocks take weeks or months to fade, which is
exactly the behaviour the ACF of squared residuals displayed. For equity
indices, estimates typically land between 0.95 and 1.00. Where this series
lands is the first thing to check in the fit below.




```python
spec_garch_n = dict(mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
res_garch11 = arch_model(returns_scaled, **spec_garch_n).fit(disp='off')

print(res_garch11.summary())
```

                         Constant Mean - GARCH Model Results                      
    ==============================================================================
    Dep. Variable:            log_returns   R-squared:                       0.000
    Mean Model:             Constant Mean   Adj. R-squared:                  0.000
    Vol Model:                      GARCH   Log-Likelihood:               -8739.51
    Distribution:                  Normal   AIC:                           17487.0
    Method:            Maximum Likelihood   BIC:                           17514.1
                                            No. Observations:                 6441
    Date:                Wed, Oct 07 2026   Df Residuals:                     6440
    Time:                        10:45:06   Df Model:                            1
                                     Mean Model                                 
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    mu             0.0630  1.003e-02      6.283  3.319e-10 [4.334e-02,8.265e-02]
                                  Volatility Model                              
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    omega          0.0256  4.843e-03      5.277  1.314e-07 [1.606e-02,3.505e-02]
    alpha[1]       0.1194  1.170e-02     10.203  1.932e-24   [9.647e-02,  0.142]
    beta[1]        0.8597  1.257e-02     68.386      0.000     [  0.835,  0.884]
    ============================================================================
    
    Covariance estimator: robust
    


```python
register_model('GARCH(1,1) — Normal', res_garch11, spec_garch_n)
show_model_table()
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
      <th>log_likelihood</th>
      <th>aic</th>
      <th>bic</th>
      <th>n_params</th>
      <th>delta_aic</th>
      <th>delta_ll</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GARCH(1,1) — Normal</th>
      <td>-8739.512405</td>
      <td>17487.024811</td>
      <td>17514.106567</td>
      <td>4.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ARCH(1) — Normal</th>
      <td>-9812.146411</td>
      <td>19630.292822</td>
      <td>19650.604140</td>
      <td>3.0</td>
      <td>2143.268012</td>
      <td>-1072.634006</td>
    </tr>
  </tbody>
</table>
</div>



```python
alpha_g = res_garch11.params['alpha[1]']
beta_g = res_garch11.params['beta[1]']
persistence = alpha_g + beta_g
half_life = np.log(0.5) / np.log(persistence)

std_resid_g11 = res_garch11.std_resid.dropna()
arch_test_resid_g11 = het_arch(std_resid_g11, nlags=20)
lb_sq_g11 = acorr_ljungbox(std_resid_g11 ** 2, lags=[20], return_df=True)
lb_stat_g11 = lb_sq_g11['lb_stat'].iloc[0]
lb_p_g11 = lb_sq_g11['lb_pvalue'].iloc[0]

print(f'alpha[1]: {alpha_g:.4f}')
print(f'beta[1]: {beta_g:.4f}')
print(f'Persistence (alpha + beta): {persistence:.4f}')
print(f'Volatility half-life: {half_life:.1f} trading days')
print(f'ARCH-LM on standardised residuals: LM = {arch_test_resid_g11[0]:.2f}, '
      f'p = {arch_test_resid_g11[1]:.6f}')
print(f'Ljung-Box on squared std residuals (lag 20): '
      f'Q = {lb_stat_g11:.2f}, p = {lb_p_g11:.6f}')

display(Markdown(f"""
#### Interpreting GARCH(1,1)

Persistence is alpha + beta = {persistence:.4f}. A shock to volatility decays
with a half-life of {half_life:.1f} trading days — roughly
{half_life/21:.1f} trading months. This is the long memory ARCH(1) could not
represent, delivered by one additional parameter. The half-life follows from
geometry: under GARCH(1,1) a variance shock decays at rate alpha + beta per
day, so the half-life is the number of trading days for the shock to lose
half its magnitude, log(0.5) / log(alpha + beta).

The same sum carries a structural condition. alpha + beta < 1 is the
covariance-stationarity requirement: below 1, conditional variance
mean-reverts to a finite long-run level, omega / (1 - alpha - beta); at or
above 1, shocks never fully decay and the unconditional variance is
undefined. {'The condition holds here.' if persistence < 1 else 'This estimate sits at or above the boundary, which needs attention before any forecast from this model is trusted.'}

The division of labour between the two coefficients is informative.
Beta ({beta_g:.4f}) dominates: most of today's variance is inherited from
yesterday's variance estimate. Alpha ({alpha_g:.4f}) is the reactive
component, the weight placed on yesterday's squared shock. Volatility under
this model is a slow-moving state occasionally jolted by news, not a fresh
draw each day.

The standardised residuals now return
LM = {arch_test_resid_g11[0]:,.2f}
(p {'< 0.001' if arch_test_resid_g11[1] < 0.001 else f'= {arch_test_resid_g11[1]:.4f}'}) on the ARCH-LM test,
corroborated by the Ljung-Box test on squared residuals
(Q = {lb_stat_g11:,.2f},
p {'< 0.001' if lb_p_g11 < 0.001 else f'= {lb_p_g11:.4f}'}).
{'The variance dynamics are captured; no significant clustering remains in either test.' if arch_test_resid_g11[1] >= 0.05 and lb_p_g11 >= 0.05 else 'Some clustering survives, though far less than under ARCH(1) — worth noting but not disqualifying.'}

What the Normal-innovation fit cannot fix is the distribution itself. The
model assumes standardised residuals are Gaussian; the data said otherwise
in Notebook 02 (excess kurtosis {kurt_nb02:.4f}). That mismatch distorts the
likelihood and, through it, the parameter estimates. The next section
replaces the Normal with Student's t and measures what changes.
"""))
```

    alpha[1]: 0.1194
    beta[1]: 0.8597
    Persistence (alpha + beta): 0.9791
    Volatility half-life: 32.9 trading days
    ARCH-LM on standardised residuals: LM = 21.68, p = 0.357861
    Ljung-Box on squared std residuals (lag 20): Q = 21.01, p = 0.396551
    



#### Interpreting GARCH(1,1)

Persistence is alpha + beta = 0.9791. A shock to volatility decays
with a half-life of 32.9 trading days — roughly
1.6 trading months. This is the long memory ARCH(1) could not
represent, delivered by one additional parameter. The half-life follows from
geometry: under GARCH(1,1) a variance shock decays at rate alpha + beta per
day, so the half-life is the number of trading days for the shock to lose
half its magnitude, log(0.5) / log(alpha + beta).

The same sum carries a structural condition. alpha + beta < 1 is the
covariance-stationarity requirement: below 1, conditional variance
mean-reverts to a finite long-run level, omega / (1 - alpha - beta); at or
above 1, shocks never fully decay and the unconditional variance is
undefined. The condition holds here.

The division of labour between the two coefficients is informative.
Beta (0.8597) dominates: most of today's variance is inherited from
yesterday's variance estimate. Alpha (0.1194) is the reactive
component, the weight placed on yesterday's squared shock. Volatility under
this model is a slow-moving state occasionally jolted by news, not a fresh
draw each day.

The standardised residuals now return
LM = 21.68
(p = 0.3579) on the ARCH-LM test,
corroborated by the Ljung-Box test on squared residuals
(Q = 21.01,
p = 0.3966).
The variance dynamics are captured; no significant clustering remains in either test.

What the Normal-innovation fit cannot fix is the distribution itself. The
model assumes standardised residuals are Gaussian; the data said otherwise
in Notebook 02 (excess kurtosis 10.6850). That mismatch distorts the
likelihood and, through it, the parameter estimates. The next section
replaces the Normal with Student's t and measures what changes.




```python
# Rescale conditional volatility back to raw return units before plotting
cond_vol_g11 = res_garch11.conditional_volatility / 100

vol_compare = pd.DataFrame({
    'realised_vol': realised_vol,
    'garch11_conditional_vol': cond_vol_g11
}).dropna()

vol_compare.plot(
    title='GARCH(1,1) Conditional Volatility vs Realised Volatility (|log return|)',
    labels={'value': 'Volatility', 'index': 'Date'}
)
```



### 5.4 GARCH(1,1), Student's t innovations


```python
display(Markdown(f"""
The GARCH(1,1) fit captured the variance dynamics but kept a distributional
assumption the data rejected in Notebook 02: excess kurtosis {kurt_nb02:.4f},
skewness {skew_nb02:.4f}, Jarque–Bera {jb_stat_nb02:,.2f}
({'p < 0.001' if jb_p_nb02 < 0.001 else f'p = {jb_p_nb02:.4f}'}). Under a Normal, a
four-sigma daily move is a once-in-decades event. This sample contains many.
When the likelihood treats observed tail events as near-impossibilities, the
optimiser compensates by distorting the variance parameters to make the tails
reachable.

Student's t innovations add one parameter: the degrees of freedom, nu,
estimated from the data. Small nu means heavy tails; as nu grows the
distribution approaches the Normal, and around nu = 30 the two are hard to
tell apart. The variance equation is unchanged from the previous fit, so any
improvement in log-likelihood and AIC is attributable to the distributional
change alone. That was the point of fitting Normal first.
"""))
```



The GARCH(1,1) fit captured the variance dynamics but kept a distributional
assumption the data rejected in Notebook 02: excess kurtosis 10.6850,
skewness -0.3487, Jarque–Bera 32,055.27
(p < 0.001). Under a Normal, a
four-sigma daily move is a once-in-decades event. This sample contains many.
When the likelihood treats observed tail events as near-impossibilities, the
optimiser compensates by distorting the variance parameters to make the tails
reachable.

Student's t innovations add one parameter: the degrees of freedom, nu,
estimated from the data. Small nu means heavy tails; as nu grows the
distribution approaches the Normal, and around nu = 30 the two are hard to
tell apart. The variance equation is unchanged from the previous fit, so any
improvement in log-likelihood and AIC is attributable to the distributional
change alone. That was the point of fitting Normal first.




```python
spec_garch_t = dict(mean='Constant', vol='GARCH', p=1, q=1, dist='t')
res_garch11_t = arch_model(returns_scaled, **spec_garch_t).fit(disp='off')

print(res_garch11_t.summary())
```

                            Constant Mean - GARCH Model Results                         
    ====================================================================================
    Dep. Variable:                  log_returns   R-squared:                       0.000
    Mean Model:                   Constant Mean   Adj. R-squared:                  0.000
    Vol Model:                            GARCH   Log-Likelihood:               -8585.33
    Distribution:      Standardized Student's t   AIC:                           17180.7
    Method:                  Maximum Likelihood   BIC:                           17214.5
                                                  No. Observations:                 6441
    Date:                      Wed, Oct 07 2026   Df Residuals:                     6440
    Time:                              10:45:07   Df Model:                            1
                                     Mean Model                                 
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    mu             0.0811  8.936e-03      9.072  1.167e-19 [6.355e-02,9.858e-02]
                                  Volatility Model                              
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    omega          0.0160  3.200e-03      5.010  5.449e-07 [9.761e-03,2.231e-02]
    alpha[1]       0.1232  1.076e-02     11.451  2.313e-30     [  0.102,  0.144]
    beta[1]        0.8708  1.037e-02     83.955      0.000     [  0.850,  0.891]
                                  Distribution                              
    ========================================================================
                     coef    std err          t      P>|t|  95.0% Conf. Int.
    ------------------------------------------------------------------------
    nu             6.1410      0.479     12.824  1.198e-37 [  5.202,  7.080]
    ========================================================================
    
    Covariance estimator: robust
    


```python
register_model("GARCH(1,1) — Student's t", res_garch11_t, spec_garch_t)
show_model_table()
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
      <th>log_likelihood</th>
      <th>aic</th>
      <th>bic</th>
      <th>n_params</th>
      <th>delta_aic</th>
      <th>delta_ll</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GARCH(1,1) — Student's t</th>
      <td>-8585.331982</td>
      <td>17180.663963</td>
      <td>17214.516159</td>
      <td>5.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>GARCH(1,1) — Normal</th>
      <td>-8739.512405</td>
      <td>17487.024811</td>
      <td>17514.106567</td>
      <td>4.0</td>
      <td>306.360848</td>
      <td>-154.180424</td>
    </tr>
    <tr>
      <th>ARCH(1) — Normal</th>
      <td>-9812.146411</td>
      <td>19630.292822</td>
      <td>19650.604140</td>
      <td>3.0</td>
      <td>2449.628859</td>
      <td>-1226.814430</td>
    </tr>
  </tbody>
</table>
</div>



```python
nu = res_garch11_t.params['nu']
alpha_t = res_garch11_t.params['alpha[1]']
beta_t = res_garch11_t.params['beta[1]']
persistence_t = alpha_t + beta_t
half_life_t = np.log(0.5) / np.log(persistence_t)

std_resid_t = res_garch11_t.std_resid.dropna()
arch_test_resid_t = het_arch(std_resid_t, nlags=20)

ll_gain = res_garch11_t.loglikelihood - res_garch11.loglikelihood
aic_drop = res_garch11.aic - res_garch11_t.aic

print(f'nu (degrees of freedom): {nu:.2f}')
print(f'alpha[1]: {alpha_t:.4f}')
print(f'beta[1]: {beta_t:.4f}')
print(f'Persistence (alpha + beta): {persistence_t:.4f}')
print(f'Volatility half-life: {half_life_t:.1f} trading days')
print(f'Log-likelihood gain vs Normal: {ll_gain:,.2f}')
print(f'AIC improvement vs Normal: {aic_drop:,.2f}')
print(f'ARCH-LM on standardised residuals: LM = {arch_test_resid_t[0]:.2f}, '
      f'p = {arch_test_resid_t[1]:.6f}')

display(Markdown(f"""
#### Interpreting the Student's t fit

The estimated degrees of freedom is nu = {nu:.2f}.
{'This is deep in heavy-tail territory: after accounting for time-varying variance, daily shocks remain far from Gaussian, and the model now says so explicitly instead of distorting other parameters to compensate.' if nu < 10 else 'Tails are moderately heavier than Gaussian — less extreme than the raw kurtosis suggested, because conditional variance absorbs part of the unconditional fat-tailedness.' if nu < 30 else 'The estimate is high enough that the fitted distribution is close to Normal, which would be unusual for daily equity returns and worth verifying.'}

The log-likelihood improves by {ll_gain:,.2f} and AIC falls by {aic_drop:,.2f}
from one added parameter. The variance equation is identical across the two
specifications, so the entire gain comes from the distributional assumption:
the heavy tails documented in Notebook 02 (excess kurtosis {kurt_nb02:.4f}) are
better represented by a Student's t than by a Gaussian. Persistence moves
from {persistence:.4f} under the Normal to {persistence_t:.4f} here — a
measure of how much the distributional misspecification was leaking into the
variance estimates.

On the ARCH-LM test the standardised residuals return
LM = {arch_test_resid_t[0]:,.2f}
(p {'< 0.001' if arch_test_resid_t[1] < 0.001 else f'= {arch_test_resid_t[1]:.4f}'}).
{'No significant clustering remains; the variance dynamics are captured.' if arch_test_resid_t[1] >= 0.05 else 'Some clustering survives at the 5% level. The specification is not perfect, and the walk-forward test below is where any practical cost will show.'}
"""))
```

    nu (degrees of freedom): 6.14
    alpha[1]: 0.1232
    beta[1]: 0.8708
    Persistence (alpha + beta): 0.9940
    Volatility half-life: 115.6 trading days
    Log-likelihood gain vs Normal: 154.18
    AIC improvement vs Normal: 306.36
    ARCH-LM on standardised residuals: LM = 23.91, p = 0.246260
    



#### Interpreting the Student's t fit

The estimated degrees of freedom is nu = 6.14.
This is deep in heavy-tail territory: after accounting for time-varying variance, daily shocks remain far from Gaussian, and the model now says so explicitly instead of distorting other parameters to compensate.

The log-likelihood improves by 154.18 and AIC falls by 306.36
from one added parameter. The variance equation is identical across the two
specifications, so the entire gain comes from the distributional assumption:
the heavy tails documented in Notebook 02 (excess kurtosis 10.6850) are
better represented by a Student's t than by a Gaussian. Persistence moves
from 0.9791 under the Normal to 0.9940 here — a
measure of how much the distributional misspecification was leaking into the
variance estimates.

On the ARCH-LM test the standardised residuals return
LM = 23.91
(p = 0.2463).
No significant clustering remains; the variance dynamics are captured.




```python
# Did the distributional fix close the loop on the heavy tails?
# NB02's locked excess kurtosis uses pandas .kurt() (bias-corrected G2).
# scipy.stats.kurtosis uses the biased Fisher estimator by default, which
# gives a different number on the same series. Using the pandas estimator here
# keeps the before/after comparison consistent with the locked figure.

kurt_raw = returns.to_frame().kurt().iloc[0]   # pandas G2, matches NB02
kurt_std = pd.Series(std_resid_t).kurt()       # same estimator

print(f'Excess kurtosis, raw returns (pandas G2):        {kurt_raw:.4f}')
print(f'Excess kurtosis, standardised residuals (G2):    {kurt_std:.4f}')
print(f'Locked NB02 value for comparison:                {kurt_nb02:.4f}')

# QQ plot of standardised residuals against the fitted standardised t
sample_q = np.sort(std_resid_t.values)
probs = (np.arange(1, len(sample_q) + 1) - 0.5) / len(sample_q)
theo_q = stats.t.ppf(probs, nu) * np.sqrt((nu - 2) / nu)

qq = pd.DataFrame({'theoretical': theo_q, 'sample': sample_q})
fig = qq.plot.scatter(
    x='theoretical', y='sample',
    title=f"QQ plot: standardised residuals vs fitted Student's t (nu = {nu:.2f})",
    labels={'theoretical': 'Theoretical quantiles', 'sample': 'Sample quantiles'}
)
lo, hi = theo_q.min(), theo_q.max()
fig.add_shape(type='line', x0=lo, y0=lo, x1=hi, y1=hi,
              line=dict(dash='dash', color='grey'))
fig
```

    Excess kurtosis, raw returns (pandas G2):        11.3407
    Excess kurtosis, standardised residuals (G2):    2.0627
    Locked NB02 value for comparison:                10.6850
    




```python
display(Markdown(f"""
#### Closing the loop on the tails

Standardisation is the model's claim made testable: if the conditional
variance path is right, dividing each return by its conditional volatility
should strip out the clustering-driven part of the fat tails. Excess kurtosis
falls from {kurt_raw:.4f} in the raw returns to {kurt_std:.4f} in the
standardised residuals (both computed with the pandas bias-corrected G2
estimator, the same estimator used for the {kurt_nb02:.4f} locked in Notebook 02;
the difference reflects the extended dataset in this notebook versus the
shorter sample NB02 was locked on).
{'Most of the unconditional fat-tailedness was volatility clustering in disguise; what remains is the genuinely heavy-tailed shock distribution the Student t is there to model.' if kurt_std < kurt_raw / 2 else 'The reduction is smaller than expected, suggesting the variance dynamics leave a substantial share of the tail behaviour unexplained — worth revisiting before trusting tail-sensitive outputs.'}

The QQ plot makes the same point graphically. Points on the dashed line mean
the fitted standardised t describes the residual quantiles well; systematic
departures in the extreme corners would mean the tails remain misrepresented
even after the distributional change.
"""))
```



#### Closing the loop on the tails

Standardisation is the model's claim made testable: if the conditional
variance path is right, dividing each return by its conditional volatility
should strip out the clustering-driven part of the fat tails. Excess kurtosis
falls from 11.3407 in the raw returns to 2.0627 in the
standardised residuals (both computed with the pandas bias-corrected G2
estimator, the same estimator used for the 10.6850 locked in Notebook 02;
the difference reflects the extended dataset in this notebook versus the
shorter sample NB02 was locked on).
Most of the unconditional fat-tailedness was volatility clustering in disguise; what remains is the genuinely heavy-tailed shock distribution the Student t is there to model.

The QQ plot makes the same point graphically. Points on the dashed line mean
the fitted standardised t describes the residual quantiles well; systematic
departures in the extreme corners would mean the tails remain misrepresented
even after the distributional change.



### 5.5 GJR-GARCH: asymmetric volatility response


```python
display(Markdown(f"""
Every model so far treats shocks symmetrically. The variance equation
responds to the squared shock, and squaring erases the sign: a -2% day and a
+2% day raise tomorrow's expected variance by the same amount. The data
disagrees with that symmetry. Skewness of {skew_nb02:.4f} was documented in
Notebook 02, and equity markets show a leverage effect: negative returns
raise future volatility more than positive returns of the same size. Falling
prices raise corporate leverage ratios, and fear propagates faster than
relief.

GJR-GARCH recovers the sign with one added parameter. An indicator switches
on when yesterday's shock was negative:

`sigma²(t) = omega + (alpha + gamma·I[eps(t-1) < 0])·eps²(t-1) + beta·sigma²(t-1)`

A positive shock feeds through with weight alpha; a negative shock with
weight alpha + gamma. If gamma is significantly positive, symmetric GARCH
understates risk after down days — exactly the days a risk system exists
for. Innovations stay Student's t, so the comparison against section 5.4
isolates the asymmetry term, the same controlled design used at every prior
step.

One accounting note: because the indicator is active on roughly half of days
under a symmetric innovation distribution, persistence for GJR is
alpha + gamma/2 + beta, not the raw coefficient sum, and the
covariance-stationarity condition applies to that adjusted sum.
"""))
```



Every model so far treats shocks symmetrically. The variance equation
responds to the squared shock, and squaring erases the sign: a -2% day and a
+2% day raise tomorrow's expected variance by the same amount. The data
disagrees with that symmetry. Skewness of -0.3487 was documented in
Notebook 02, and equity markets show a leverage effect: negative returns
raise future volatility more than positive returns of the same size. Falling
prices raise corporate leverage ratios, and fear propagates faster than
relief.

GJR-GARCH recovers the sign with one added parameter. An indicator switches
on when yesterday's shock was negative:

`sigma²(t) = omega + (alpha + gamma·I[eps(t-1) < 0])·eps²(t-1) + beta·sigma²(t-1)`

A positive shock feeds through with weight alpha; a negative shock with
weight alpha + gamma. If gamma is significantly positive, symmetric GARCH
understates risk after down days — exactly the days a risk system exists
for. Innovations stay Student's t, so the comparison against section 5.4
isolates the asymmetry term, the same controlled design used at every prior
step.

One accounting note: because the indicator is active on roughly half of days
under a symmetric innovation distribution, persistence for GJR is
alpha + gamma/2 + beta, not the raw coefficient sum, and the
covariance-stationarity condition applies to that adjusted sum.




```python
spec_gjr_t = dict(mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='t')
res_gjr = arch_model(returns_scaled, **spec_gjr_t).fit(disp='off')

print(res_gjr.summary())
```

                          Constant Mean - GJR-GARCH Model Results                       
    ====================================================================================
    Dep. Variable:                  log_returns   R-squared:                       0.000
    Mean Model:                   Constant Mean   Adj. R-squared:                  0.000
    Vol Model:                        GJR-GARCH   Log-Likelihood:               -8476.24
    Distribution:      Standardized Student's t   AIC:                           16964.5
    Method:                  Maximum Likelihood   BIC:                           17005.1
                                                  No. Observations:                 6441
    Date:                      Wed, Oct 07 2026   Df Residuals:                     6440
    Time:                              10:45:07   Df Model:                            1
                                     Mean Model                                 
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    mu             0.0531  9.031e-03      5.876  4.196e-09 [3.537e-02,7.077e-02]
                                   Volatility Model                              
    =============================================================================
                     coef    std err          t      P>|t|       95.0% Conf. Int.
    -----------------------------------------------------------------------------
    omega          0.0186  3.209e-03      5.809  6.269e-09  [1.235e-02,2.493e-02]
    alpha[1]       0.0000  9.063e-03      0.000      1.000 [-1.776e-02,1.776e-02]
    gamma[1]       0.2033  2.076e-02      9.796  1.170e-22      [  0.163,  0.244]
    beta[1]        0.8811  1.245e-02     70.786      0.000      [  0.857,  0.905]
                                  Distribution                              
    ========================================================================
                     coef    std err          t      P>|t|  95.0% Conf. Int.
    ------------------------------------------------------------------------
    nu             6.7601      0.593     11.400  4.161e-30 [  5.598,  7.922]
    ========================================================================
    
    Covariance estimator: robust
    


```python
register_model("GJR-GARCH(1,1,1) — Student's t", res_gjr, spec_gjr_t)
show_model_table()
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
      <th>log_likelihood</th>
      <th>aic</th>
      <th>bic</th>
      <th>n_params</th>
      <th>delta_aic</th>
      <th>delta_ll</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GJR-GARCH(1,1,1) — Student's t</th>
      <td>-8476.235699</td>
      <td>16964.471398</td>
      <td>17005.094033</td>
      <td>6.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>GARCH(1,1) — Student's t</th>
      <td>-8585.331982</td>
      <td>17180.663963</td>
      <td>17214.516159</td>
      <td>5.0</td>
      <td>216.192565</td>
      <td>-109.096282</td>
    </tr>
    <tr>
      <th>GARCH(1,1) — Normal</th>
      <td>-8739.512405</td>
      <td>17487.024811</td>
      <td>17514.106567</td>
      <td>4.0</td>
      <td>522.553413</td>
      <td>-263.276706</td>
    </tr>
    <tr>
      <th>ARCH(1) — Normal</th>
      <td>-9812.146411</td>
      <td>19630.292822</td>
      <td>19650.604140</td>
      <td>3.0</td>
      <td>2665.821424</td>
      <td>-1335.910712</td>
    </tr>
  </tbody>
</table>
</div>



```python
alpha_j = res_gjr.params['alpha[1]']
gamma_j = res_gjr.params['gamma[1]']
beta_j = res_gjr.params['beta[1]']
gamma_p = res_gjr.pvalues['gamma[1]']
alpha_j_p = res_gjr.pvalues['alpha[1]']
persistence_gjr = alpha_j + 0.5 * gamma_j + beta_j

aic_gjr_gain = res_garch11_t.aic - res_gjr.aic

std_resid_gjr = res_gjr.std_resid.dropna()
arch_test_resid_gjr = het_arch(std_resid_gjr, nlags=20)
lb_sq_gjr = acorr_ljungbox(std_resid_gjr ** 2, lags=[20], return_df=True)
lb_stat_gjr = lb_sq_gjr['lb_stat'].iloc[0]
lb_p_gjr = lb_sq_gjr['lb_pvalue'].iloc[0]

print(f'alpha[1]: {alpha_j:.4e} (p = {alpha_j_p:.4f})')
print(f'gamma[1]: {gamma_j:.4f} (p = {gamma_p:.2e})')
print(f'beta[1]: {beta_j:.4f}')
print(f'Persistence (alpha + gamma/2 + beta): {persistence_gjr:.4f}')
print(f'AIC change vs symmetric GARCH-t: {aic_gjr_gain:+,.2f}')
print(f'ARCH-LM on standardised residuals: LM = {arch_test_resid_gjr[0]:.2f}, '
      f'p = {arch_test_resid_gjr[1]:.6f}')
print(f'Ljung-Box on squared std residuals (lag 20): '
      f'Q = {lb_stat_gjr:.2f}, p = {lb_p_gjr:.6f}')

display(Markdown(f"""
#### Interpreting GJR-GARCH

The asymmetry parameter is gamma = {gamma_j:.4f}
(p {'< 0.001' if gamma_p < 0.001 else f'= {gamma_p:.4f}'}).
{f'The leverage effect is confirmed at the model level. A negative shock feeds into next-day variance with weight alpha + gamma = {alpha_j + gamma_j:.4f}, against {alpha_j:.4e} for a positive shock of the same size. The sign of a move carries information the squared shock alone discards, consistent with the negative skewness ({skew_nb02:.4f}) documented in Notebook 02.' if gamma_p < 0.05 and gamma_j > 0 else 'The asymmetry term is not significant at the 5% level. The leverage effect visible in the unconditional skewness does not survive as a conditional-variance mechanism in this sample, and the added parameter is not justified.'}

{f'AIC improves by {aic_gjr_gain:,.2f} over the symmetric Student t fit.' if aic_gjr_gain > 0 else f'AIC worsens by {-aic_gjr_gain:,.2f} relative to the symmetric Student t fit: whatever asymmetry exists does not pay for its parameter.'}
For the regime classifier downstream, this matters most in Stress and Crisis:
an asymmetric model re-rates risk upward faster after drawdowns, which is
when the classification is consequential.

#### Alpha at the boundary

Alpha pins at {alpha_j:.2e}, effectively zero — the lower bound of its
parameter space. The p-value of {alpha_j_p:.4f} is an artifact of this
boundary solution, not evidence that alpha is truly zero in the
data-generating process. Standard Wald standard errors do not have their
usual properties at a parameter constraint, so inference on alpha requires
caution.

The economic implication is that positive shocks contribute nothing to
next-day variance through the ARCH channel. All shock-driven variance comes
from the gamma term, active only on negative days. This is a strong claim,
consistent with the leverage effect but more extreme than most equity-index
estimates in the literature. It survives the AIC comparison, so the
specification earns its place, but the boundary deserves honest reporting.
The Ljung-Box test on squared standardised residuals returns
Q = {lb_stat_gjr:,.2f}
(p {'< 0.001' if lb_p_gjr < 0.001 else f'= {lb_p_gjr:.4f}'}),
{'confirming no residual serial dependence in variance.' if lb_p_gjr >= 0.05 else 'with some residual serial dependence surviving.'}
"""))
```

    alpha[1]: 0.0000e+00 (p = 1.0000)
    gamma[1]: 0.2033 (p = 1.17e-22)
    beta[1]: 0.8811
    Persistence (alpha + gamma/2 + beta): 0.9827
    AIC change vs symmetric GARCH-t: +216.19
    ARCH-LM on standardised residuals: LM = 18.44, p = 0.558185
    Ljung-Box on squared std residuals (lag 20): Q = 17.16, p = 0.642337
    



#### Interpreting GJR-GARCH

The asymmetry parameter is gamma = 0.2033
(p < 0.001).
The leverage effect is confirmed at the model level. A negative shock feeds into next-day variance with weight alpha + gamma = 0.2033, against 0.0000e+00 for a positive shock of the same size. The sign of a move carries information the squared shock alone discards, consistent with the negative skewness (-0.3487) documented in Notebook 02.

AIC improves by 216.19 over the symmetric Student t fit.
For the regime classifier downstream, this matters most in Stress and Crisis:
an asymmetric model re-rates risk upward faster after drawdowns, which is
when the classification is consequential.

#### Alpha at the boundary

Alpha pins at 0.00e+00, effectively zero — the lower bound of its
parameter space. The p-value of 1.0000 is an artifact of this
boundary solution, not evidence that alpha is truly zero in the
data-generating process. Standard Wald standard errors do not have their
usual properties at a parameter constraint, so inference on alpha requires
caution.

The economic implication is that positive shocks contribute nothing to
next-day variance through the ARCH channel. All shock-driven variance comes
from the gamma term, active only on negative days. This is a strong claim,
consistent with the leverage effect but more extreme than most equity-index
estimates in the literature. It survives the AIC comparison, so the
specification earns its place, but the boundary deserves honest reporting.
The Ljung-Box test on squared standardised residuals returns
Q = 17.16
(p = 0.6423),
confirming no residual serial dependence in variance.



#### Asymmetric shocks: skewed Student's t innovations

The GJR equation lets negative shocks raise next-day variance more than
positive ones, but its Student's t still treats the shocks themselves as
symmetric: a fall and a rise of the same standardised size are equally
likely. The next cell tests that assumption on the symmetric fit above.

This candidate keeps the GJR variance equation and replaces the Student's t
with Hansen's skewed Student's t, which adds one parameter, lambda, for
asymmetry. A negative lambda means large falls are more likely than large
rises of the same size. It was added in October 2026, after NB08's in-sample
VaR backtest rejected the symmetric specification at both levels, so it
enters the same AIC comparison as every other specification rather than
replacing the incumbent by hand.

One accounting change follows. Under a skewed distribution, the GJR
indicator is active on a share of days equal to P(Z < 0), which need not be
one half, so persistence becomes alpha + gamma·P(Z < 0) + beta.


```python
# Under a symmetric distribution, residuals beyond the 0.5% cutoff on each
# side should split evenly. Count each side in the symmetric fit above.
z_sym = res_gjr.std_resid.dropna()
tail_cut = float(res_gjr.model.distribution.ppf(0.995, res_gjr.params[['nu']]))
n_tail_low = int((z_sym < -tail_cut).sum())
n_tail_high = int((z_sym > tail_cut).sum())
n_tail_expected = 0.005 * len(z_sym)
split_p = binomtest(n_tail_low, n_tail_low + n_tail_high, 0.5).pvalue

if split_p < 0.05:
    side = 'falling' if n_tail_low > n_tail_high else 'rising'
    tail_lean_text = (f"The extreme shocks lean to the {side} side, which a "
                      "symmetric distribution cannot represent.")
else:
    tail_lean_text = "This sample shows no significant lean in the extreme shocks."

display(Markdown(f"""
In the symmetric fit, {n_tail_low} standardised residuals fall below the lower
0.5% cutoff and {n_tail_high} rise above the upper one, with the cutoffs at
±{tail_cut:.3f} and {n_tail_expected:.1f} days expected on each side. An exact
binomial test of an even split gives p = {split_p:.2e}. {tail_lean_text}
"""))
```



In the symmetric fit, 49 standardised residuals fall below the lower
0.5% cutoff and 6 rise above the upper one, with the cutoffs at
±2.972 and 32.2 days expected on each side. An exact
binomial test of an even split gives p = 1.82e-09. The extreme shocks lean to the falling side, which a symmetric distribution cannot represent.




```python
spec_gjr_skewt = dict(mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='skewt')
res_gjr_skewt = arch_model(returns_scaled, **spec_gjr_skewt).fit(disp='off')
print(res_gjr_skewt.summary())
```

                             Constant Mean - GJR-GARCH Model Results                         
    =========================================================================================
    Dep. Variable:                       log_returns   R-squared:                       0.000
    Mean Model:                        Constant Mean   Adj. R-squared:                  0.000
    Vol Model:                             GJR-GARCH   Log-Likelihood:               -8438.20
    Distribution:      Standardized Skew Student's t   AIC:                           16890.4
    Method:                       Maximum Likelihood   BIC:                           16937.8
                                                       No. Observations:                 6441
    Date:                           Wed, Oct 07 2026   Df Residuals:                     6440
    Time:                                   10:45:07   Df Model:                            1
                                     Mean Model                                 
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    mu             0.0278  9.613e-03      2.897  3.767e-03 [9.008e-03,4.669e-02]
                                   Volatility Model                              
    =============================================================================
                     coef    std err          t      P>|t|       95.0% Conf. Int.
    -----------------------------------------------------------------------------
    omega          0.0198  3.281e-03      6.028  1.659e-09  [1.335e-02,2.621e-02]
    alpha[1]       0.0000  8.805e-03      0.000      1.000 [-1.726e-02,1.726e-02]
    gamma[1]       0.2089  2.148e-02      9.723  2.414e-22      [  0.167,  0.251]
    beta[1]        0.8801  1.269e-02     69.378      0.000      [  0.855,  0.905]
                                  Distribution                              
    ========================================================================
                     coef    std err          t      P>|t|  95.0% Conf. Int.
    ------------------------------------------------------------------------
    eta            7.4861      0.702     10.671  1.395e-26 [  6.111,  8.861]
    lambda        -0.1496  1.617e-02     -9.246  2.335e-20 [ -0.181, -0.118]
    ========================================================================
    
    Covariance estimator: robust
    


```python
register_model("GJR-GARCH(1,1,1) — skewed Student's t", res_gjr_skewt, spec_gjr_skewt)
show_model_table()
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
      <th>log_likelihood</th>
      <th>aic</th>
      <th>bic</th>
      <th>n_params</th>
      <th>delta_aic</th>
      <th>delta_ll</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GJR-GARCH(1,1,1) — skewed Student's t</th>
      <td>-8438.198075</td>
      <td>16890.396149</td>
      <td>16937.789223</td>
      <td>7.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>GJR-GARCH(1,1,1) — Student's t</th>
      <td>-8476.235699</td>
      <td>16964.471398</td>
      <td>17005.094033</td>
      <td>6.0</td>
      <td>74.075249</td>
      <td>-38.037625</td>
    </tr>
    <tr>
      <th>GARCH(1,1) — Student's t</th>
      <td>-8585.331982</td>
      <td>17180.663963</td>
      <td>17214.516159</td>
      <td>5.0</td>
      <td>290.267814</td>
      <td>-147.133907</td>
    </tr>
    <tr>
      <th>GARCH(1,1) — Normal</th>
      <td>-8739.512405</td>
      <td>17487.024811</td>
      <td>17514.106567</td>
      <td>4.0</td>
      <td>596.628662</td>
      <td>-301.314331</td>
    </tr>
    <tr>
      <th>ARCH(1) — Normal</th>
      <td>-9812.146411</td>
      <td>19630.292822</td>
      <td>19650.604140</td>
      <td>3.0</td>
      <td>2739.896673</td>
      <td>-1373.948337</td>
    </tr>
  </tbody>
</table>
</div>



```python
lam = res_gjr_skewt.params['lambda']
lam_p = res_gjr_skewt.pvalues['lambda']
d_aic = res_gjr.aic - res_gjr_skewt.aic

if lam_p >= 0.05:
    skew_text = "not significantly different from zero, so the shocks show no clear lean"
elif lam < 0:
    skew_text = ("a lean to the left: large falls are more likely than large "
                 "rises of the same size")
else:
    skew_text = ("a lean to the right: large rises are more likely than large "
                 "falls of the same size")
aic_winner = 'skewed' if d_aic > 0 else 'symmetric'

display(Markdown(f"""
The skew parameter is lambda = {lam:.4f} (p = {lam_p:.2e}), {skew_text}.
Against the symmetric GJR specification, AIC differs by {abs(d_aic):,.2f}
points in favour of the {aic_winner} model. By the Burnham and Anderson
guideline, a gap above 10 leaves essentially no support for the trailing
model.
"""))
```



The skew parameter is lambda = -0.1496 (p = 2.34e-20), a lean to the left: large falls are more likely than large rises of the same size.
Against the symmetric GJR specification, AIC differs by 74.08
points in favour of the skewed model. By the Burnham and Anderson
guideline, a gap above 10 leaves essentially no support for the trailing
model.



### 5.6 In-sample comparison and model selection


```python
show_model_table()

comparison = pd.DataFrame(model_results).T.sort_values('aic')
best_label = comparison.index[0]
best_spec = model_specs[best_label]
best_res = model_objects[best_label]

# The runner-up is whichever specification ranks second by AIC, so the label
# stays correct on every rerun. It sits the out-of-sample test in section 6.
runnerup_label = comparison.index[1]

print(f'Selected specification: {best_label}')
print(f'Runner-up:              {runnerup_label}')

display(Markdown(f"""
The table ranks all {len(comparison)} fits by AIC; **{best_label}** leads and
advances to forecast evaluation, the regime classifier, and the risk summary,
with **{runnerup_label}** in second place. Two caveats keep this honest. These
are in-sample statistics computed on the same data the models were fitted to,
and the AIC comparison is valid only because every model was fitted on the
same scaled series, so the rescaling constant cancels. The delta columns
restate the ranking in relative terms: delta_aic is each model's AIC distance
from the leader, and by the Burnham and Anderson guideline a distance above
10 means essentially no support for the trailing model.

In-sample fit does not establish forecasting ability. That question belongs
to the next section, which evaluates the winning specification against both
benchmarks on data none of them has seen.
"""))
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
      <th>log_likelihood</th>
      <th>aic</th>
      <th>bic</th>
      <th>n_params</th>
      <th>delta_aic</th>
      <th>delta_ll</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GJR-GARCH(1,1,1) — skewed Student's t</th>
      <td>-8438.198075</td>
      <td>16890.396149</td>
      <td>16937.789223</td>
      <td>7.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>GJR-GARCH(1,1,1) — Student's t</th>
      <td>-8476.235699</td>
      <td>16964.471398</td>
      <td>17005.094033</td>
      <td>6.0</td>
      <td>74.075249</td>
      <td>-38.037625</td>
    </tr>
    <tr>
      <th>GARCH(1,1) — Student's t</th>
      <td>-8585.331982</td>
      <td>17180.663963</td>
      <td>17214.516159</td>
      <td>5.0</td>
      <td>290.267814</td>
      <td>-147.133907</td>
    </tr>
    <tr>
      <th>GARCH(1,1) — Normal</th>
      <td>-8739.512405</td>
      <td>17487.024811</td>
      <td>17514.106567</td>
      <td>4.0</td>
      <td>596.628662</td>
      <td>-301.314331</td>
    </tr>
    <tr>
      <th>ARCH(1) — Normal</th>
      <td>-9812.146411</td>
      <td>19630.292822</td>
      <td>19650.604140</td>
      <td>3.0</td>
      <td>2739.896673</td>
      <td>-1373.948337</td>
    </tr>
  </tbody>
</table>
</div>


    Selected specification: GJR-GARCH(1,1,1) — skewed Student's t
    Runner-up:              GJR-GARCH(1,1,1) — Student's t
    



The table ranks all 5 fits by AIC; **GJR-GARCH(1,1,1) — skewed Student's t** leads and
advances to forecast evaluation, the regime classifier, and the risk summary,
with **GJR-GARCH(1,1,1) — Student's t** in second place. Two caveats keep this honest. These
are in-sample statistics computed on the same data the models were fitted to,
and the AIC comparison is valid only because every model was fitted on the
same scaled series, so the rescaling constant cancels. The delta columns
restate the ranking in relative terms: delta_aic is each model's AIC distance
from the leader, and by the Burnham and Anderson guideline a distance above
10 means essentially no support for the trailing model.

In-sample fit does not establish forecasting ability. That question belongs
to the next section, which evaluates the winning specification against both
benchmarks on data none of them has seen.



## 6. Forecast evaluation


```python
display(Markdown(f"""
Section 5 answered which specification best explains the data. This section
answers a different question: which forecaster predicts unseen observations
best? In-sample statistics reward fitting the past. The claim that matters is
predictive — could this model have forecast tomorrow's volatility using only
information available today? Walk-forward evaluation answers that directly.

### Walk-forward design

- Test window: the final 252 trading days, matching the evaluation convention
  from Notebook 04.
- Refit schedule: the model is re-estimated every 21 trading days on an
  expanding window. Between refits, parameters stay fixed while the
  conditional variance updates daily with each new observation.
- Forecast: one-step-ahead conditional volatility, divided by 100 to return
  to raw units.
- Forecasters: the AIC winner ({best_label}) and the runner-up
  ({runnerup_label}), both evaluated against persistence and EWMA.
  Running the runner-up confirms the in-sample ranking holds out of sample.

Every volatility model here forecasts sigma, the conditional standard
deviation, while the realised target is the absolute return. The expected
absolute value of a shock is c × sigma, where c = E|Z| depends on the
innovation distribution: sqrt(2/pi) ≈ 0.7979 for a Normal, roughly 0.75 to
0.80 for a standardised Student's t at the nu values typical of equity
indices, and for a skewed Student's t a value computed from its fitted shape
parameters. Scoring raw sigma against |r| would penalise a forecaster for a
unit mismatch rather than for forecasting skill.

Each loss function therefore receives every forecaster in the units it is
built for. RMSE and MAE compare expected absolute returns against |r|: GARCH
forecasts use the factor implied by each refit's fitted distribution, and
EWMA uses the Normal factor its RiskMetrics design assumes. QLIKE compares
variances against squared returns, so every forecaster enters it as
unconverted sigma. Persistence forecasts yesterday's absolute return directly
and needs no conversion. The unconverted sigma errors are reported alongside,
to show the size of the unit effect.
"""))
```



Section 5 answered which specification best explains the data. This section
answers a different question: which forecaster predicts unseen observations
best? In-sample statistics reward fitting the past. The claim that matters is
predictive — could this model have forecast tomorrow's volatility using only
information available today? Walk-forward evaluation answers that directly.

### Walk-forward design

- Test window: the final 252 trading days, matching the evaluation convention
  from Notebook 04.
- Refit schedule: the model is re-estimated every 21 trading days on an
  expanding window. Between refits, parameters stay fixed while the
  conditional variance updates daily with each new observation.
- Forecast: one-step-ahead conditional volatility, divided by 100 to return
  to raw units.
- Forecasters: the AIC winner (GJR-GARCH(1,1,1) — skewed Student's t) and the runner-up
  (GJR-GARCH(1,1,1) — Student's t), both evaluated against persistence and EWMA.
  Running the runner-up confirms the in-sample ranking holds out of sample.

Every volatility model here forecasts sigma, the conditional standard
deviation, while the realised target is the absolute return. The expected
absolute value of a shock is c × sigma, where c = E|Z| depends on the
innovation distribution: sqrt(2/pi) ≈ 0.7979 for a Normal, roughly 0.75 to
0.80 for a standardised Student's t at the nu values typical of equity
indices, and for a skewed Student's t a value computed from its fitted shape
parameters. Scoring raw sigma against |r| would penalise a forecaster for a
unit mismatch rather than for forecasting skill.

Each loss function therefore receives every forecaster in the units it is
built for. RMSE and MAE compare expected absolute returns against |r|: GARCH
forecasts use the factor implied by each refit's fitted distribution, and
EWMA uses the Normal factor its RiskMetrics design assumes. QLIKE compares
variances against squared returns, so every forecaster enters it as
unconverted sigma. Persistence forecasts yesterday's absolute return directly
and needs no conversion. The unconverted sigma errors are reported alongside,
to show the size of the unit effect.




```python

def abs_return_factor(params):
    """E|Z| for the fitted innovation distribution.

    Skewed Student's t when 'eta' and 'lambda' are present; Student's t
    (standardised, nu degrees of freedom) when 'nu' is present; Normal
    otherwise.
    """
    if 'eta' in params and 'lambda' in params:
        # Z has mean zero, so E|Z| = -2 * E[Z; Z < 0], the first lower
        # partial moment of the fitted distribution at zero.
        shape = [params['eta'], params['lambda']]
        return -2 * SkewStudent().partial_moment(1, 0.0, shape)
    if 'nu' in params:
        nu_ = params['nu']
        return (2 * np.sqrt(nu_ - 2) * gamma_fn((nu_ + 1) / 2)
                / (np.sqrt(np.pi) * (nu_ - 1) * gamma_fn(nu_ / 2)))
    return np.sqrt(2 / np.pi)

test_size = 252
refit_every = 21
n = len(returns_scaled)
test_start = n - test_size

# Runner-up specification for OOS confirmation
runnerup_spec = model_specs[runnerup_label]

wf_rows = []
params_wf = None
factor_wf = None
params_ru = None
factor_ru = None

for i in range(test_size):
    t = test_start + i
    history = returns_scaled.iloc[:t]

    if i % refit_every == 0:
        # Refit winner
        res_refit = arch_model(history, **best_spec).fit(disp='off')
        params_wf = res_refit.params
        factor_wf = abs_return_factor(params_wf)
        # Refit runner-up
        res_refit_ru = arch_model(history, **runnerup_spec).fit(disp='off')
        params_ru = res_refit_ru.params
        factor_ru = abs_return_factor(params_ru)

    # Winner: fixed parameters, updated information set
    fixed = arch_model(history, **best_spec).fix(np.asarray(params_wf))
    f = fixed.forecast(horizon=1, reindex=False)
    sigma_raw = np.sqrt(f.variance.values[-1, 0]) / 100

    # Runner-up
    fixed_ru = arch_model(history, **runnerup_spec).fix(np.asarray(params_ru))
    f_ru = fixed_ru.forecast(horizon=1, reindex=False)
    sigma_raw_ru = np.sqrt(f_ru.variance.values[-1, 0]) / 100

    wf_rows.append({
        'date': returns_scaled.index[t],
        'sigma_forecast': sigma_raw,
        'abs_return_forecast': sigma_raw * factor_wf,
        'sigma_forecast_ru': sigma_raw_ru,
        'abs_return_forecast_ru': sigma_raw_ru * factor_ru,
    })

wf = pd.DataFrame(wf_rows).set_index('date')
wf['realised_vol'] = realised_vol.reindex(wf.index)
wf['persistence_forecast'] = realised_vol.shift(1).reindex(wf.index)

# EWMA benchmark on the test window (expanding from the same history).
# Its sigma feeds QLIKE; the converted forecast feeds RMSE and MAE.
ewma_var_full = (returns ** 2).ewm(alpha=(1 - ewma_lambda), adjust=False).mean()
wf['ewma_sigma'] = np.sqrt(ewma_var_full.shift(1)).reindex(wf.index)
wf['ewma_forecast'] = wf['ewma_sigma'] * ewma_abs_factor

wf = wf.dropna()

print(f'Winner:    {best_label}')
print(f'Runner-up: {runnerup_label}')
print(f'Walk-forward window: {wf.index.min().date()} to {wf.index.max().date()}')
print(f'Forecasts produced: {len(wf)}')
```

    Winner:    GJR-GARCH(1,1,1) — skewed Student's t
    Runner-up: GJR-GARCH(1,1,1) — Student's t
    Walk-forward window: 2025-09-25 to 2026-09-25
    Forecasts produced: 252
    


```python
# ── Error metrics ──
# RMSE and MAE score expected absolute returns against |r|. QLIKE is defined
# on variance forecasts, so every forecaster enters it as unconverted sigma.
def qlike(actual, forecast):
    """QLIKE loss: mean of (actual^2 / forecast^2 - log(actual^2 / forecast^2) - 1).
    Patton (2011) consistent loss for volatility proxy evaluation. `forecast`
    is a conditional standard deviation, so forecast^2 is the variance
    forecast the loss is defined on."""
    ratio = (actual ** 2) / (forecast ** 2)
    return np.mean(ratio - np.log(ratio) - 1)

rmse_garch_wf = root_mean_squared_error(wf['realised_vol'], wf['abs_return_forecast'])
mae_garch_wf = mean_absolute_error(wf['realised_vol'], wf['abs_return_forecast'])
qlike_garch_wf = qlike(wf['realised_vol'], wf['sigma_forecast'])
rmse_sigma_wf = root_mean_squared_error(wf['realised_vol'], wf['sigma_forecast'])

rmse_ru_wf = root_mean_squared_error(wf['realised_vol'], wf['abs_return_forecast_ru'])
mae_ru_wf = mean_absolute_error(wf['realised_vol'], wf['abs_return_forecast_ru'])
qlike_ru_wf = qlike(wf['realised_vol'], wf['sigma_forecast_ru'])

rmse_pers_wf = root_mean_squared_error(wf['realised_vol'], wf['persistence_forecast'])
mae_pers_wf = mean_absolute_error(wf['realised_vol'], wf['persistence_forecast'])
qlike_pers_wf = np.nan  # QLIKE divides by forecast², undefined when persistence ≈ 0

rmse_ewma_wf = root_mean_squared_error(wf['realised_vol'], wf['ewma_forecast'])
mae_ewma_wf = mean_absolute_error(wf['realised_vol'], wf['ewma_forecast'])
qlike_ewma_wf = qlike(wf['realised_vol'], wf['ewma_sigma'])
rmse_ewma_sigma_wf = root_mean_squared_error(wf['realised_vol'], wf['ewma_sigma'])

improvement_vs_pers = (rmse_pers_wf - rmse_garch_wf) / rmse_pers_wf * 100
improvement_vs_ewma = (rmse_ewma_wf - rmse_garch_wf) / rmse_ewma_wf * 100
ru_gap_pct = (rmse_ru_wf - rmse_garch_wf) / rmse_garch_wf * 100

ewma_label = f'EWMA λ={ewma_lambda}'
forecast_comparison = pd.DataFrame({
    'RMSE': [rmse_pers_wf, rmse_ewma_wf, rmse_garch_wf, rmse_ru_wf,
             rmse_ewma_sigma_wf, rmse_sigma_wf],
    'MAE': [mae_pers_wf, mae_ewma_wf, mae_garch_wf, mae_ru_wf, np.nan, np.nan],
    'QLIKE': [qlike_pers_wf, qlike_ewma_wf, qlike_garch_wf, qlike_ru_wf,
              np.nan, np.nan],
}, index=[
    'Persistence (test window)',
    ewma_label,
    best_label,
    runnerup_label,
    f'{ewma_label} (raw sigma, unit reference)',
    f'{best_label} (raw sigma, unit reference)',
])
display(forecast_comparison)

print(f'RMSE change vs persistence: {improvement_vs_pers:+.1f}%')
print(f'RMSE change vs EWMA:        {improvement_vs_ewma:+.1f}%')
print(f'Runner-up RMSE vs winner:   {ru_gap_pct:+.2f}%')

# ── Diebold-Mariano tests under squared-error loss ──
# Newey-West HAC variance for serial correlation in the loss differential.
e_garch = (wf['realised_vol'] - wf['abs_return_forecast']).values
e_pers = (wf['realised_vol'] - wf['persistence_forecast']).values
e_ewma = (wf['realised_vol'] - wf['ewma_forecast']).values

d_vs_pers = e_pers ** 2 - e_garch ** 2   # positive = GARCH better
d_vs_ewma = e_ewma ** 2 - e_garch ** 2   # positive = GARCH better

def dm_test_nw(d, max_lag=None):
    """Diebold-Mariano test statistic with Newey-West HAC variance."""
    n = len(d)
    if max_lag is None:
        max_lag = int(np.ceil(n ** (1/3)))
    d_bar = d.mean()
    # Newey-West HAC variance
    gamma_0 = np.mean((d - d_bar) ** 2)
    gamma_sum = 0
    for h in range(1, max_lag + 1):
        w = 1 - h / (max_lag + 1)  # Bartlett kernel
        gamma_h = np.mean((d[h:] - d_bar) * (d[:-h] - d_bar))
        gamma_sum += 2 * w * gamma_h
    var_d = (gamma_0 + gamma_sum) / n
    dm_stat = d_bar / np.sqrt(var_d) if var_d > 0 else np.inf
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

dm_stat_pers, dm_p_pers = dm_test_nw(d_vs_pers)
dm_stat_ewma, dm_p_ewma = dm_test_nw(d_vs_ewma)

# ── Diebold-Mariano under QLIKE loss: the selected model vs EWMA ──
def qlike_loss(actual, forecast):
    """Per-observation QLIKE loss (not averaged); forecast is a sigma."""
    ratio = (actual ** 2) / (forecast ** 2)
    return ratio - np.log(ratio) - 1

ql_garch = qlike_loss(wf['realised_vol'].values, wf['sigma_forecast'].values)
ql_ewma = qlike_loss(wf['realised_vol'].values, wf['ewma_sigma'].values)
d_qlike_ewma = ql_ewma - ql_garch  # positive = GARCH better under QLIKE

dm_stat_qlike_ewma, dm_p_qlike_ewma = dm_test_nw(d_qlike_ewma)

print(f'\nDiebold-Mariano (Newey-West HAC):')
print(f'  vs Persistence (SE loss): DM = {dm_stat_pers:.3f}, p = {dm_p_pers:.4f}')
print(f'  vs EWMA (SE loss):        DM = {dm_stat_ewma:.3f}, p = {dm_p_ewma:.4f}')
print(f'  vs EWMA (QLIKE loss):     DM = {dm_stat_qlike_ewma:.3f}, p = {dm_p_qlike_ewma:.4f}')

# ── Sentences that depend on the results ──
def p_text(p):
    return '< 0.001' if p < 0.001 else f'= {p:.4f}'

def change_text(pct):
    return f"{abs(pct):.1f}% {'lower' if pct > 0 else 'higher'}"

if rmse_ewma_wf < rmse_pers_wf * 0.9:
    ewma_vs_pers_text = ("a large improvement over persistence, confirming that most of "
                         "the persistence benchmark's weakness is its reliance on a single "
                         "noisy observation rather than smooth decay.")
elif rmse_ewma_wf < rmse_pers_wf:
    ewma_vs_pers_text = "a moderate improvement over persistence."
else:
    ewma_vs_pers_text = "no improvement over persistence in this window."

if improvement_vs_pers > 0 and improvement_vs_ewma > 0:
    garch_vs_bench_text = ("The model beats both benchmarks, including the tougher EWMA "
                           "bar, with every forecast scored in the same units.")
elif improvement_vs_pers > 0:
    garch_vs_bench_text = ("The model beats persistence but not EWMA on RMSE. With every "
                           "forecast scored in the same units, EWMA already captures most "
                           "of the forecastable structure in this window.")
elif improvement_vs_ewma > 0:
    garch_vs_bench_text = ("The model beats EWMA but not persistence on RMSE, an unusual "
                           "ordering worth checking before relying on either result.")
else:
    garch_vs_bench_text = ("The model beats neither benchmark on RMSE over this window, a "
                           "null result reported as found.")

ru_gap_text = (f"{abs(ru_gap_pct):.2f}%" if abs(ru_gap_pct) >= 0.005
               else "less than 0.01%")
if rmse_garch_wf < rmse_ru_wf:
    ru_text = (f"The in-sample AIC ranking holds out of sample: the runner-up's RMSE is "
               f"{ru_gap_text} higher than the winner's.")
else:
    ru_text = (f"The runner-up matches or beats the winner on RMSE, by {ru_gap_text}, "
               f"so the in-sample AIC advantage does not translate into better point "
               f"forecasts over this window. The AIC-selected specification is "
               f"retained for consistency.")

spec_diff = {k for k in set(best_spec) | set(runnerup_spec)
             if best_spec.get(k) != runnerup_spec.get(k)}
if spec_diff == {'dist'}:
    ru_text += (" The two specifications share a variance equation and differ only in "
                "the innovation distribution, which mainly shapes the tails. Typical-sized "
                "moves dominate this test, so a small gap is expected either way; the "
                "tails are tested by the VaR backtest in Notebook 08.")

if dm_p_pers < 0.05 and dm_stat_pers > 0:
    dm_pers_text = "a statistically significant improvement."
elif dm_p_pers < 0.05:
    dm_pers_text = "a statistically significant deterioration: persistence forecasts better."
else:
    dm_pers_text = ("not significant at the 5% level. The 252-day window is one draw, "
                    "and the point estimate should not be over-interpreted.")

if dm_p_ewma < 0.05 and dm_stat_ewma > 0:
    dm_ewma_text = "a significant edge over the industry-standard naive forecast."
elif dm_p_ewma < 0.05:
    dm_ewma_text = "a significant disadvantage: EWMA forecasts better under squared-error loss."
else:
    dm_ewma_text = "not significant at the 5% level against the tougher benchmark."

qlike_scores = {ewma_label: qlike_ewma_wf, best_label: qlike_garch_wf,
                runnerup_label: qlike_ru_wf}
qlike_order = sorted(qlike_scores, key=qlike_scores.get)
qlike_order_text = ', then '.join(f'{name} ({qlike_scores[name]:.4f})'
                                  for name in qlike_order)

if dm_p_qlike_ewma < 0.05:
    qlike_dm_text = ("a significant difference in favour of "
                     + ("the selected model." if dm_stat_qlike_ewma > 0 else "EWMA."))
else:
    qlike_dm_text = ("not significant at the 5% level: under QLIKE the two forecasts "
                     "cannot be separated in this window.")

if (improvement_vs_ewma > 0) == (dm_stat_qlike_ewma > 0):
    loss_agreement_text = "Squared-error loss and QLIKE agree on the direction."
else:
    loss_agreement_text = "Squared-error loss and QLIKE disagree on the direction."

display(Markdown(f"""
### Walk-forward results

Both benchmarks and both GARCH specifications were evaluated on the same
{len(wf)}-day test window. QLIKE (Patton 2011) is reported alongside RMSE and
MAE because it remains a consistent loss function even when the volatility
proxy is noisy, a property squared-error loss does not have. RMSE and MAE
score expected absolute returns; QLIKE scores each forecaster's sigma, the
units it is defined on.

**Persistence** produced RMSE = {rmse_pers_wf:.6f}.
**EWMA (λ = {ewma_lambda})** produced RMSE = {rmse_ewma_wf:.6f}, {ewma_vs_pers_text}
**{best_label}** produced RMSE = {rmse_garch_wf:.6f}, {change_text(improvement_vs_pers)}
than persistence and {change_text(improvement_vs_ewma)} than EWMA.
{garch_vs_bench_text}

**{runnerup_label}** produced RMSE = {rmse_ru_wf:.6f}. {ru_text}

The Diebold-Mariano test (Newey-West HAC variance, squared-error loss) asks
whether each RMSE difference is distinguishable from noise. Against
persistence, DM = {dm_stat_pers:.3f} (p {p_text(dm_p_pers)}): {dm_pers_text}
Against EWMA, DM = {dm_stat_ewma:.3f} (p {p_text(dm_p_ewma)}): {dm_ewma_text}

Under QLIKE, where lower is better, the ordering is {qlike_order_text}.
QLIKE is asymmetric: it penalises a variance forecast that comes in below the
realised squared return more heavily than one that comes in above it. A
Diebold-Mariano test of the selected model against EWMA under QLIKE loss
returns DM = {dm_stat_qlike_ewma:.3f} (p = {dm_p_qlike_ewma:.4f}):
{qlike_dm_text} {loss_agreement_text} Persistence is excluded from QLIKE
because its near-zero forecasts on flat days make the loss undefined.

For reference, scoring raw sigma against |r| gives RMSE =
{rmse_sigma_wf:.6f} for {best_label} and {rmse_ewma_sigma_wf:.6f} for EWMA.
The gap between each forecaster's two rows is the unit effect the conversion
removes, not a difference in forecasting skill.
"""))
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
      <th>RMSE</th>
      <th>MAE</th>
      <th>QLIKE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Persistence (test window)</th>
      <td>0.007562</td>
      <td>0.005686</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>EWMA λ=0.94</th>
      <td>0.005366</td>
      <td>0.004080</td>
      <td>1.696274</td>
    </tr>
    <tr>
      <th>GJR-GARCH(1,1,1) — skewed Student's t</th>
      <td>0.005236</td>
      <td>0.004005</td>
      <td>1.635379</td>
    </tr>
    <tr>
      <th>GJR-GARCH(1,1,1) — Student's t</th>
      <td>0.005233</td>
      <td>0.004000</td>
      <td>1.634800</td>
    </tr>
    <tr>
      <th>EWMA λ=0.94 (raw sigma, unit reference)</th>
      <td>0.005694</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>GJR-GARCH(1,1,1) — skewed Student's t (raw sigma, unit reference)</th>
      <td>0.005719</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    RMSE change vs persistence: +30.8%
    RMSE change vs EWMA:        +2.4%
    Runner-up RMSE vs winner:   -0.06%
    
    Diebold-Mariano (Newey-West HAC):
      vs Persistence (SE loss): DM = 5.510, p = 0.0000
      vs EWMA (SE loss):        DM = 1.768, p = 0.0770
      vs EWMA (QLIKE loss):     DM = 1.617, p = 0.1059
    



### Walk-forward results

Both benchmarks and both GARCH specifications were evaluated on the same
252-day test window. QLIKE (Patton 2011) is reported alongside RMSE and
MAE because it remains a consistent loss function even when the volatility
proxy is noisy, a property squared-error loss does not have. RMSE and MAE
score expected absolute returns; QLIKE scores each forecaster's sigma, the
units it is defined on.

**Persistence** produced RMSE = 0.007562.
**EWMA (λ = 0.94)** produced RMSE = 0.005366, a large improvement over persistence, confirming that most of the persistence benchmark's weakness is its reliance on a single noisy observation rather than smooth decay.
**GJR-GARCH(1,1,1) — skewed Student's t** produced RMSE = 0.005236, 30.8% lower
than persistence and 2.4% lower than EWMA.
The model beats both benchmarks, including the tougher EWMA bar, with every forecast scored in the same units.

**GJR-GARCH(1,1,1) — Student's t** produced RMSE = 0.005233. The runner-up matches or beats the winner on RMSE, by 0.06%, so the in-sample AIC advantage does not translate into better point forecasts over this window. The AIC-selected specification is retained for consistency. The two specifications share a variance equation and differ only in the innovation distribution, which mainly shapes the tails. Typical-sized moves dominate this test, so a small gap is expected either way; the tails are tested by the VaR backtest in Notebook 08.

The Diebold-Mariano test (Newey-West HAC variance, squared-error loss) asks
whether each RMSE difference is distinguishable from noise. Against
persistence, DM = 5.510 (p < 0.001): a statistically significant improvement.
Against EWMA, DM = 1.768 (p = 0.0770): not significant at the 5% level against the tougher benchmark.

Under QLIKE, where lower is better, the ordering is GJR-GARCH(1,1,1) — Student's t (1.6348), then GJR-GARCH(1,1,1) — skewed Student's t (1.6354), then EWMA λ=0.94 (1.6963).
QLIKE is asymmetric: it penalises a variance forecast that comes in below the
realised squared return more heavily than one that comes in above it. A
Diebold-Mariano test of the selected model against EWMA under QLIKE loss
returns DM = 1.617 (p = 0.1059):
not significant at the 5% level: under QLIKE the two forecasts cannot be separated in this window. Squared-error loss and QLIKE agree on the direction. Persistence is excluded from QLIKE
because its near-zero forecasts on flat days make the loss undefined.

For reference, scoring raw sigma against |r| gives RMSE =
0.005719 for GJR-GARCH(1,1,1) — skewed Student's t and 0.005694 for EWMA.
The gap between each forecaster's two rows is the unit effect the conversion
removes, not a difference in forecasting skill.




```python
wf[['realised_vol', 'abs_return_forecast', 'ewma_forecast', 'persistence_forecast']].plot(
    title='Walk-forward one-day-ahead volatility forecasts (252-day test window)',
    labels={'value': 'Volatility', 'index': 'Date'}
)
```



## 7. Volatility regimes


```python
display(Markdown(f"""
A point forecast answers how much. Risk management also needs a categorical
answer: what kind of market is this? Regime classification compresses the
conditional volatility path into four states that map directly to decisions
such as position sizing and capital deployment schedules.

The regimes are defined on the annualised conditional volatility from the
selected model ({best_label}), using percentile thresholds:

- Calm: below the 25th percentile
- Normal: 25th to 75th percentile
- Stress: 75th to 95th percentile
- Crisis: above the 95th percentile

Thresholds are computed once on the full sample, which makes the regime
shares mechanical — roughly 25/50/20/5 by construction. The classifier is
therefore not judged on how often each regime occurs. It is judged on *when*:
the labels must land on the right episodes. Two known stress periods serve as
validation cases below, the COVID crash of early 2020 and the 2022 rate-hike
cycle. A production system would freeze thresholds on a training window; for
this descriptive validation, full-sample percentiles are sufficient.
"""))
```



A point forecast answers how much. Risk management also needs a categorical
answer: what kind of market is this? Regime classification compresses the
conditional volatility path into four states that map directly to decisions
such as position sizing and capital deployment schedules.

The regimes are defined on the annualised conditional volatility from the
selected model (GJR-GARCH(1,1,1) — skewed Student's t), using percentile thresholds:

- Calm: below the 25th percentile
- Normal: 25th to 75th percentile
- Stress: 75th to 95th percentile
- Crisis: above the 95th percentile

Thresholds are computed once on the full sample, which makes the regime
shares mechanical — roughly 25/50/20/5 by construction. The classifier is
therefore not judged on how often each regime occurs. It is judged on *when*:
the labels must land on the right episodes. Two known stress periods serve as
validation cases below, the COVID crash of early 2020 and the 2022 rate-hike
cycle. A production system would freeze thresholds on a training window; for
this descriptive validation, full-sample percentiles are sufficient.




```python
# Annualised conditional volatility from the selected model, in raw units
cond_vol = best_res.conditional_volatility / 100
cond_vol_ann = cond_vol * np.sqrt(252)

q25, q75, q95 = cond_vol_ann.quantile([0.25, 0.75, 0.95])

regimes = pd.cut(
    cond_vol_ann,
    bins=[-np.inf, q25, q75, q95, np.inf],
    labels=['Calm', 'Normal', 'Stress', 'Crisis']
)

regime_df = pd.DataFrame({'cond_vol_ann': cond_vol_ann, 'regime': regimes})

print(f'Thresholds (annualised volatility):')
print(f'  Calm   < {q25:.4f}')
print(f'  Normal < {q75:.4f}')
print(f'  Stress < {q95:.4f}')
print(f'  Crisis >= {q95:.4f}')
print()
print(regime_df['regime'].value_counts())
```

    Thresholds (annualised volatility):
      Calm   < 0.1018
      Normal < 0.1931
      Stress < 0.3379
      Crisis >= 0.3379
    
    regime
    Normal    3220
    Calm      1611
    Stress    1288
    Crisis     322
    Name: count, dtype: int64
    


```python
covid = regime_df.loc['2020-02-15':'2020-04-30', 'regime']
covid_share = covid.value_counts(normalize=True)

hikes = regime_df.loc['2022-01-01':'2022-12-31', 'regime']
hikes_share = hikes.value_counts(normalize=True)

print('COVID window (2020-02-15 to 2020-04-30):')
print(covid_share.round(3))
print()
print('Rate-hike year (2022):')
print(hikes_share.round(3))
```

    COVID window (2020-02-15 to 2020-04-30):
    regime
    Crisis    0.846
    Normal    0.096
    Stress    0.058
    Calm      0.000
    Name: proportion, dtype: float64
    
    Rate-hike year (2022):
    regime
    Stress    0.578
    Normal    0.315
    Crisis    0.100
    Calm      0.008
    Name: proportion, dtype: float64
    


```python
covid_crisis = covid_share.get('Crisis', 0.0)
covid_stress_plus = covid_crisis + covid_share.get('Stress', 0.0)
hikes_stress_plus = hikes_share.get('Stress', 0.0) + hikes_share.get('Crisis', 0.0)

display(Markdown(f"""
### Validation against known stress episodes

The unconditional Crisis share is 5% by construction. In the COVID window
(2020-02-15 to 2020-04-30), the Crisis share is {covid_crisis:.0%}, with
{covid_stress_plus:.0%} of days at Stress or worse.
{"The classifier concentrates its rarest label exactly where the market's fastest drawdown on record occurred." if covid_crisis > 0.5 else "The concentration is weaker than expected for the fastest drawdown on record — worth investigating before this classifier feeds any downstream signal."}

The 2022 rate-hike year is a different kind of test: chronic elevated
volatility rather than an acute spike. {hikes_stress_plus:.0%} of 2022
trading days classify as Stress or Crisis, against an unconditional 25%.
{"The classifier reads 2022 as persistently elevated, matching the character of that year." if hikes_stress_plus > 0.35 else "That is close to the unconditional share, suggesting 2022's stress registered less in conditional volatility than the narrative implies. The numbers, not the narrative, are the record."}

Together the two windows test both failure modes that matter for a risk
system: missing an acute crisis and missing a chronic one.
"""))
```



### Validation against known stress episodes

The unconditional Crisis share is 5% by construction. In the COVID window
(2020-02-15 to 2020-04-30), the Crisis share is 85%, with
90% of days at Stress or worse.
The classifier concentrates its rarest label exactly where the market's fastest drawdown on record occurred.

The 2022 rate-hike year is a different kind of test: chronic elevated
volatility rather than an acute spike. 68% of 2022
trading days classify as Stress or Crisis, against an unconditional 25%.
The classifier reads 2022 as persistently elevated, matching the character of that year.

Together the two windows test both failure modes that matter for a risk
system: missing an acute crisis and missing a chronic one.




```python
fig = cond_vol_ann.plot(
    title='Annualised conditional volatility with regime thresholds',
    labels={'value': 'Annualised volatility', 'index': 'Date'}
)
fig.add_hline(y=q25, line_dash='dash', annotation_text='Calm / Normal')
fig.add_hline(y=q75, line_dash='dash', annotation_text='Normal / Stress')
fig.add_hline(y=q95, line_dash='dash', annotation_text='Stress / Crisis')
fig
```



## 8. Risk intelligence summary


```python
display(Markdown("""
The final output assembles what a portfolio or risk decision needs each day:
the next-day volatility forecast in both of its units, its position in the
historical distribution, and the regime label. This is decision support, not
a trading signal. The system says nothing about direction — Notebook 04
established that direction is not forecastable here. What it quantifies is
the risk environment those decisions are made in.
"""))
```



The final output assembles what a portfolio or risk decision needs each day:
the next-day volatility forecast in both of its units, its position in the
historical distribution, and the regime label. This is decision support, not
a trading signal. The system says nothing about direction — Notebook 04
established that direction is not forecastable here. What it quantifies is
the risk environment those decisions are made in.




```python
# Next-day forecast from the selected model's full-sample fit
final_forecast = best_res.forecast(horizon=1, reindex=False)
sigma_next = np.sqrt(final_forecast.variance.values[-1, 0]) / 100
sigma_next_ann = sigma_next * np.sqrt(252)
abs_move_next = sigma_next * abs_return_factor(best_res.params)

if sigma_next_ann < q25:
    regime_next = 'Calm'
elif sigma_next_ann < q75:
    regime_next = 'Normal'
elif sigma_next_ann < q95:
    regime_next = 'Stress'
else:
    regime_next = 'Crisis'

vol_percentile = (cond_vol_ann < sigma_next_ann).mean()

regime_context = {
    'Calm': 'Volatility sits in the bottom quartile of its historical distribution.',
    'Normal': 'Volatility sits within its typical historical range.',
    'Stress': 'Volatility is elevated, above the 75th percentile of its history.',
    'Crisis': 'Volatility is extreme, above the 95th percentile of its history.',
}

as_of = returns.index.max().date()

display(Markdown(f"""
### Risk intelligence summary — as of {as_of}

| Quantity | Value |
|---|---|
| Next-day conditional standard deviation | {sigma_next:.4%} daily ({sigma_next_ann:.2%} annualised) |
| Expected absolute daily move | {abs_move_next:.4%} |
| Historical percentile | {vol_percentile:.0%} |
| Regime | **{regime_next}** |

{regime_context[regime_next]} The forecast is produced by {best_label},
fitted on the full sample, and the regime label applies the thresholds
validated in section 7.

The two forecast rows are different units of the same forecast, and the
distinction matters for anyone translating these numbers into potential
losses. The conditional standard deviation is the model's sigma: the input to
VaR-style calculations, where a loss threshold is sigma scaled by a quantile
of the fitted innovation distribution. The expected absolute daily move is
c × sigma, the expectation of tomorrow's |return| — a typical move, not a
bad one. Turning the {sigma_next_ann:.1%} annualised figure into a potential
daily loss uses the standard deviation row and a chosen confidence level,
never the absolute-move row.
"""))
```



### Risk intelligence summary — as of 2026-09-25

| Quantity | Value |
|---|---|
| Next-day conditional standard deviation | 0.6384% daily (10.13% annualised) |
| Expected absolute daily move | 0.4871% |
| Historical percentile | 24% |
| Regime | **Calm** |

Volatility sits in the bottom quartile of its historical distribution. The forecast is produced by GJR-GARCH(1,1,1) — skewed Student's t,
fitted on the full sample, and the regime label applies the thresholds
validated in section 7.

The two forecast rows are different units of the same forecast, and the
distinction matters for anyone translating these numbers into potential
losses. The conditional standard deviation is the model's sigma: the input to
VaR-style calculations, where a loss threshold is sigma scaled by a quantile
of the fitted innovation distribution. The expected absolute daily move is
c × sigma, the expectation of tomorrow's |return| — a typical move, not a
bad one. Turning the 10.1% annualised figure into a potential
daily loss uses the standard deviation row and a chosen confidence level,
never the absolute-move row.



## 9. What this notebook established


```python
# Sentences that depend on this run's results
if lam_p >= 0.05:
    skew_summary = (f"A skewed Student's t then tested whether the shocks themselves "
                    f"lean to one side; the skew parameter was not significant "
                    f"(lambda = {lam:.4f}, p = {lam_p:.2e}).")
else:
    lean_side = 'falls' if lam < 0 else 'rises'
    skew_summary = (f"A skewed Student's t then tested whether the shocks themselves "
                    f"lean to one side: lambda = {lam:.4f} (p = {lam_p:.2e}), so large "
                    f"{lean_side} are more likely than moves of the same size in the "
                    f"other direction.")
skew_summary += (f" Against the symmetric GJR fit, AIC differed by {abs(d_aic):,.2f} "
                 f"points in favour of the {aic_winner} model.")

if rmse_garch_wf < rmse_ru_wf:
    ru_summary = 'confirming the in-sample AIC ranking holds'
else:
    ru_summary = ('with no clear out-of-sample advantage for the winner, which is '
                  'retained as the AIC-selected model')

qlike_verdict = 'significant' if dm_p_qlike_ewma < 0.05 else 'not significant'
qlike_summary = (f"Under QLIKE, scored on each forecaster's sigma, the ordering was "
                 f"{qlike_order_text}; the test of the selected model against EWMA "
                 f"returned DM = {dm_stat_qlike_ewma:.3f}, p = {dm_p_qlike_ewma:.4f}, "
                 f"{qlike_verdict} at the 5% level.")

test_vol_ann = returns.loc[wf.index].std() * np.sqrt(252)
full_vol_ann = returns.std() * np.sqrt(252)
test_calmer = 'calmer' if test_vol_ann < full_vol_ann else 'more volatile'
test_window_text = (f"This test window was {test_calmer} than the sample as a whole, "
                    f"with annualised volatility of {test_vol_ann:.1%} against "
                    f"{full_vol_ann:.1%} over the full sample.")

display(Markdown(f"""
The ARCH-LM test (LM = {arch_test[0]:,.2f}, p ≈ 0) confirmed conditional
heteroskedasticity in daily returns. ARCH(1) validated the mechanism but not
the memory: significant ARCH effects survived in its standardised residuals
(both ARCH-LM and Ljung-Box on squared residuals). GARCH(1,1) fixed that
with one parameter. The distributional change from Normal to Student's t
(nu = {nu:.2f}) improved AIC by {aic_drop:,.2f} with the variance equation
held fixed, isolating the value of modelling the tails honestly.
{f"GJR-GARCH then confirmed the leverage effect: gamma = {gamma_j:.4f}, so negative shocks raise next-day variance with weight {alpha_j + gamma_j:.4f} against {alpha_j:.4e} for positive shocks of the same size. Alpha pinning at its lower bound means all shock-driven variance flows through the asymmetry channel, a boundary result flagged in section 5.5." if gamma_p < 0.05 and gamma_j > 0 else f"The GJR asymmetry term was tested and found non-significant: the unconditional skewness of {skew_nb02:.4f} does not translate into a conditional-variance asymmetry in this sample, a null result reported as found."}
{skew_summary}
The AIC winner, {best_label}, advanced to every downstream section.

Out of sample, two benchmarks were tested rather than one: persistence
(single-lag absolute return) and EWMA (RiskMetrics λ = {ewma_lambda}), the
industry-standard naive forecast. {best_label}
{f"beat persistence by {improvement_vs_pers:.1f}% on RMSE (DM p {'< 0.001' if dm_p_pers < 0.001 else f'= {dm_p_pers:.4f}'}) and {'also beat EWMA by ' + f'{improvement_vs_ewma:.1f}%' + ' (DM p ' + ('<0.001' if dm_p_ewma < 0.001 else f'= {dm_p_ewma:.4f}') + ')' if improvement_vs_ewma > 0 else 'did not beat EWMA on RMSE, meaning EWMA already captures most forecastable structure in this window'} over the final 252 trading days" if improvement_vs_pers > 0 else f"did not beat the persistence benchmark over the final 252 trading days ({improvement_vs_pers:+.1f}% RMSE change), a null result reported as found"}.
The runner-up ({runnerup_label}) was also evaluated out of sample
({ru_summary}).
QLIKE (Patton 2011) was reported alongside RMSE and MAE as a
proxy-consistent loss function. {qlike_summary}

The regime classifier's validation against the 2020 COVID crash and the 2022
rate-hike cycle is reported in section 7, and the notebook closes with the
daily risk intelligence summary the Daily Market Risk Report will consume —
stated in both of its units, conditional standard deviation and expected
absolute move.

### Limitations

This evaluation uses a single test window of 252 trading days. That mirrors
the Notebook 04 convention and keeps every model on identical footing, but
one window is one draw from one market regime, and forecasting performance
can differ under others. {test_window_text} The Diebold-Mariano tests
quantify whether the observed differences are distinguishable from noise
within this window, but they do not guarantee the ranking generalises. An
extended study would repeat the walk-forward over multiple rolling or
expanding windows and report the distribution of outcomes. The regime
thresholds share a related caveat, noted in section 7: they are computed on
the full sample for descriptive validation, where a production system would
freeze them on a training window.

### Evaluation protocol for Notebook 06

Notebook 06 inherits the core of this protocol: the same realised-volatility
target (absolute daily log returns), the same 252-day walk-forward window,
the same 21-day refit schedule, the persistence benchmark, and RMSE, MAE and
Diebold-Mariano tests. Its networks forecast the absolute return directly, so
they need no unit conversion. It does not recompute EWMA or QLIKE, so its
comparisons with the selected model rest on RMSE and MAE. Any deep learning
result reported there is directly comparable to the RMSE and MAE figures
above, so a difference between the two reflects the models, not the test.
"""))
```



The ARCH-LM test (LM = 1,867.15, p ≈ 0) confirmed conditional
heteroskedasticity in daily returns. ARCH(1) validated the mechanism but not
the memory: significant ARCH effects survived in its standardised residuals
(both ARCH-LM and Ljung-Box on squared residuals). GARCH(1,1) fixed that
with one parameter. The distributional change from Normal to Student's t
(nu = 6.14) improved AIC by 306.36 with the variance equation
held fixed, isolating the value of modelling the tails honestly.
GJR-GARCH then confirmed the leverage effect: gamma = 0.2033, so negative shocks raise next-day variance with weight 0.2033 against 0.0000e+00 for positive shocks of the same size. Alpha pinning at its lower bound means all shock-driven variance flows through the asymmetry channel, a boundary result flagged in section 5.5.
A skewed Student's t then tested whether the shocks themselves lean to one side: lambda = -0.1496 (p = 2.34e-20), so large falls are more likely than moves of the same size in the other direction. Against the symmetric GJR fit, AIC differed by 74.08 points in favour of the skewed model.
The AIC winner, GJR-GARCH(1,1,1) — skewed Student's t, advanced to every downstream section.

Out of sample, two benchmarks were tested rather than one: persistence
(single-lag absolute return) and EWMA (RiskMetrics λ = 0.94), the
industry-standard naive forecast. GJR-GARCH(1,1,1) — skewed Student's t
beat persistence by 30.8% on RMSE (DM p < 0.001) and also beat EWMA by 2.4% (DM p = 0.0770) over the final 252 trading days.
The runner-up (GJR-GARCH(1,1,1) — Student's t) was also evaluated out of sample
(with no clear out-of-sample advantage for the winner, which is retained as the AIC-selected model).
QLIKE (Patton 2011) was reported alongside RMSE and MAE as a
proxy-consistent loss function. Under QLIKE, scored on each forecaster's sigma, the ordering was GJR-GARCH(1,1,1) — Student's t (1.6348), then GJR-GARCH(1,1,1) — skewed Student's t (1.6354), then EWMA λ=0.94 (1.6963); the test of the selected model against EWMA returned DM = 1.617, p = 0.1059, not significant at the 5% level.

The regime classifier's validation against the 2020 COVID crash and the 2022
rate-hike cycle is reported in section 7, and the notebook closes with the
daily risk intelligence summary the Daily Market Risk Report will consume —
stated in both of its units, conditional standard deviation and expected
absolute move.

### Limitations

This evaluation uses a single test window of 252 trading days. That mirrors
the Notebook 04 convention and keeps every model on identical footing, but
one window is one draw from one market regime, and forecasting performance
can differ under others. This test window was calmer than the sample as a whole, with annualised volatility of 13.0% against 19.1% over the full sample. The Diebold-Mariano tests
quantify whether the observed differences are distinguishable from noise
within this window, but they do not guarantee the ranking generalises. An
extended study would repeat the walk-forward over multiple rolling or
expanding windows and report the distribution of outcomes. The regime
thresholds share a related caveat, noted in section 7: they are computed on
the full sample for descriptive validation, where a production system would
freeze them on a training window.

### Evaluation protocol for Notebook 06

Notebook 06 inherits the core of this protocol: the same realised-volatility
target (absolute daily log returns), the same 252-day walk-forward window,
the same 21-day refit schedule, the persistence benchmark, and RMSE, MAE and
Diebold-Mariano tests. Its networks forecast the absolute return directly, so
they need no unit conversion. It does not recompute EWMA or QLIKE, so its
comparisons with the selected model rest on RMSE and MAE. Any deep learning
result reported there is directly comparable to the RMSE and MAE figures
above, so a difference between the two reflects the models, not the test.




```python
for name, obj in list(globals().items()):
    if type(obj).__name__ == 'ARCHModelResult':
        print(name)
```

    res_arch1
    res_garch11
    res_garch11_t
    res_gjr
    res_gjr_skewt
    best_res
    res_refit
    res_refit_ru
    


```python
# ── Export metrics for downstream notebooks ──
metrics_path = Path('../data/locked_metrics.json')
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

metrics['notebook_05'] = {
    'best_label':              best_label,
    'runnerup_label':          runnerup_label,
    'wf_rmse_garch':           float(rmse_garch_wf),
    'wf_mae_garch':            float(mae_garch_wf),
    'wf_rmse_runnerup':        float(rmse_ru_wf),
    'wf_rmse_persistence':     float(rmse_pers_wf),
    'wf_mae_persistence':      float(mae_pers_wf),
    'wf_rmse_ewma':            float(rmse_ewma_wf),
    'wf_mae_ewma':             float(mae_ewma_wf),
    'improvement_rmse_pct':    float(improvement_vs_pers),
    'improvement_vs_ewma_pct': float(improvement_vs_ewma),
    'dm_stat_vs_persistence':  float(dm_stat_pers),
    'dm_p_vs_persistence':     float(dm_p_pers),
    'dm_stat_vs_ewma':         float(dm_stat_ewma),
    'dm_p_vs_ewma':            float(dm_p_ewma),
    'qlike_garch':             float(qlike_garch_wf),
    'qlike_ewma':              float(qlike_ewma_wf),
    'dm_stat_vs_ewma_qlike':   float(dm_stat_qlike_ewma),
    'dm_p_vs_ewma_qlike':      float(dm_p_qlike_ewma),
    'ewma_lambda':             float(ewma_lambda),
    'ewma_abs_factor':         float(ewma_abs_factor),
}

# Fitted parameters of the selected model, read from whichever specification
# won, so no downstream notebook has to assume a distribution.
p = best_res.params
dist = best_res.model.distribution
shape_names = list(dist.parameter_names())
shape = [float(p[name]) for name in shape_names]
a = sum(v for k, v in p.items() if k.startswith('alpha'))
b = sum(v for k, v in p.items() if k.startswith('beta'))
g = sum(v for k, v in p.items() if k.startswith('gamma'))
# The GJR indicator is active on a share of days equal to P(Z < 0): one half
# under a symmetric distribution, a fitted value under a skewed one.
p_neg = float(dist.cdf(0.0, shape)) if shape_names else 0.5

metrics['notebook_05'].update({
    'garch_dist':        best_spec['dist'],
    'garch_shape':       dict(zip(shape_names, shape)),
    'garch_p_neg':       p_neg,
    'garch_mu':          float(p['mu']),
    'garch_alpha':       float(a),
    'garch_gamma':       float(g),
    'garch_beta':        float(b),
    'garch_persistence': float(a + b + g * p_neg),
    'garch_nobs':        int(best_res.nobs),
    # Estimated parameters excluding the constant mean, the count Notebook 06
    # quotes when comparing model sizes.
    'garch_n_params':    int(len(p) - 1),
})

metrics_path.write_text(json.dumps(metrics, indent=2))

print(f'Exported notebook_05 metrics to {metrics_path.resolve()}')
for k, v in metrics['notebook_05'].items():
    print(f'  {k}: {v}')
```

    Exported notebook_05 metrics to C:\Users\Mena\Documents\Python\sp500-market-intelligence\data\locked_metrics.json
      best_label: GJR-GARCH(1,1,1) — skewed Student's t
      runnerup_label: GJR-GARCH(1,1,1) — Student's t
      wf_rmse_garch: 0.005236451386802539
      wf_mae_garch: 0.004004899081652212
      wf_rmse_runnerup: 0.0052333492427069535
      wf_rmse_persistence: 0.0075617841947451326
      wf_mae_persistence: 0.005685889011073224
      wf_rmse_ewma: 0.005366311915722928
      wf_mae_ewma: 0.0040804419879624685
      improvement_rmse_pct: 30.751113071416707
      improvement_vs_ewma_pct: 2.4199213716948993
      dm_stat_vs_persistence: 5.509968834327064
      dm_p_vs_persistence: 3.5889723859483524e-08
      dm_stat_vs_ewma: 1.7684927735850506
      dm_p_vs_ewma: 0.0769785592278942
      qlike_garch: 1.6353786665435341
      qlike_ewma: 1.6962735076059883
      dm_stat_vs_ewma_qlike: 1.6170685933042492
      dm_p_vs_ewma_qlike: 0.10586347533729468
      ewma_lambda: 0.94
      ewma_abs_factor: 0.7978845608028654
      garch_dist: skewt
      garch_shape: {'eta': 7.4860859169257195, 'lambda': -0.14955043971231133}
      garch_p_neg: 0.4730010391405909
      garch_mu: 0.027849516594884077
      garch_alpha: 0.0
      garch_gamma: 0.20888460958582072
      garch_beta: 0.8801426246017412
      garch_persistence: 0.978945261996311
      garch_nobs: 6441
      garch_n_params: 6
    
