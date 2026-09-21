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
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error
from IPython.display import display, Markdown
import json
from pathlib import Path
from scipy import stats
from scipy.special import gamma as gamma_fn
from scipy.stats import t as t_dist

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

    Notebook 02 — skewness: -0.3493, excess kurtosis: 10.6566, Jarque-Bera: 31,748.52
    Notebook 04 — ARIMA RMSE: 0.01064, historical-mean RMSE: 0.01064
    


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
(RMSE 0.01064
in both cases),
a null result consistent with weak-form market efficiency. The residual
diagnostics revealed structure in the *variance*: fat tails (excess kurtosis
10.6566), negative skewness (-0.3493), and persistent
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




    (6412, 64)




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
<p>5 rows × 64 columns</p>
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

    Observations: 6,412
    Mean daily return: 0.000276
    Daily volatility (std): 0.012057
    Annualized volatility: 0.1914 (19.14%)
    


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



The dataset contains 6,412 daily observations spanning more than
25 years, including periods of extreme stress (2008, 2020) and
extended low-volatility regimes.

Daily volatility (1.21%) is roughly 44
times the mean daily return (0.028%). This imbalance is
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
# exponentially weighted moving average of squared returns, with no refit and
# no distributional assumption. This is a much tougher bar than persistence,
# because EWMA already models smooth decay — the same structural advantage
# GARCH formalises with estimated parameters.
ewma_lambda = 0.94
sq_returns = returns ** 2
ewma_var = sq_returns.ewm(alpha=(1 - ewma_lambda), adjust=False).mean()
benchmark['ewma_forecast'] = np.sqrt(ewma_var.shift(1))  # E[sigma] as sqrt(variance)

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

    Persistence Benchmark  RMSE: 0.010787  MAE: 0.007120
    EWMA (λ=0.94)       RMSE: 0.008399  MAE: 0.006072
    


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
squared returns — the same smooth-decay structure GARCH formalises, but with
a fixed decay parameter and no distributional assumption. Over the full
sample it produced RMSE = {rmse_ewma:.6f} and MAE = {mae_ewma:.6f}.

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
RMSE = 0.010787 and MAE = 0.007120.

**EWMA (RiskMetrics, λ = 0.94)** is the industry-standard naive
forecast. Conditional variance is an exponentially weighted moving average of
squared returns — the same smooth-decay structure GARCH formalises, but with
a fixed decay parameter and no distributional assumption. Over the full
sample it produced RMSE = 0.008399 and MAE = 0.006072.

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

    LM Statistic: 1857.96
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
test returned **LM = 1,857.96**
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

Four specifications are fitted in a controlled sequence, each changing one
thing: ARCH(1) tests the mechanism, GARCH(1,1) adds memory, Student's t fixes
the tails, and GJR-GARCH tests asymmetry. Each model registers itself in a
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
simplification and a known misspecification: excess kurtosis of 10.6566 and
skewness of -0.3493 were documented in Notebook 02. The Normal assumption is
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
(LM = 1,857.96,
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
    Vol Model:                       ARCH   Log-Likelihood:               -9778.96
    Distribution:                  Normal   AIC:                           19563.9
    Method:            Maximum Likelihood   BIC:                           19584.2
                                            No. Observations:                 6412
    Date:                Tue, Aug 18 2026   Df Residuals:                     6411
    Time:                        23:37:16   Df Model:                            1
                                     Mean Model                                 
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    mu             0.0581  1.581e-02      3.678  2.349e-04 [2.716e-02,8.913e-02]
                                Volatility Model                            
    ========================================================================
                     coef    std err          t      P>|t|  95.0% Conf. Int.
    ------------------------------------------------------------------------
    omega          0.9112  4.758e-02     19.151  9.485e-82 [  0.818,  1.004]
    alpha[1]       0.4017  5.506e-02      7.296  2.955e-13 [  0.294,  0.510]
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
      <td>-9778.957602</td>
      <td>19563.915203</td>
      <td>19584.212983</td>
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

    omega: 0.9112
    alpha[1]: 0.4017 (p = 2.95e-13)
    ARCH-LM on standardised residuals: LM = 1083.87, p = 0.000000
    Ljung-Box on squared std residuals (lag 20): Q = 2456.94, p = 0.000000
    



#### Interpreting ARCH(1)

The single ARCH coefficient is alpha[1] = 0.4017
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
LM = 1,083.87
(p < 0.001).
Significant conditional heteroskedasticity survives the fit. One lag of squared shocks is not enough.

The Ljung-Box test on squared standardised residuals corroborates
(Q = 2,456.94,
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
    Vol Model:                      GARCH   Log-Likelihood:               -8711.00
    Distribution:                  Normal   AIC:                           17430.0
    Method:            Maximum Likelihood   BIC:                           17457.1
                                            No. Observations:                 6412
    Date:                Tue, Aug 18 2026   Df Residuals:                     6411
    Time:                        23:37:16   Df Model:                            1
                                     Mean Model                                 
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    mu             0.0634  1.007e-02      6.297  3.032e-10 [4.366e-02,8.311e-02]
                                  Volatility Model                              
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    omega          0.0256  4.863e-03      5.264  1.411e-07 [1.607e-02,3.513e-02]
    alpha[1]       0.1195  1.174e-02     10.178  2.479e-24   [9.647e-02,  0.142]
    beta[1]        0.8597  1.261e-02     68.203      0.000     [  0.835,  0.884]
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
      <td>-8711.002062</td>
      <td>17430.004123</td>
      <td>17457.067829</td>
      <td>4.0</td>
      <td>0.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>ARCH(1) — Normal</th>
      <td>-9778.957602</td>
      <td>19563.915203</td>
      <td>19584.212983</td>
      <td>3.0</td>
      <td>2133.91108</td>
      <td>-1067.95554</td>
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

    alpha[1]: 0.1195
    beta[1]: 0.8597
    Persistence (alpha + beta): 0.9792
    Volatility half-life: 33.0 trading days
    ARCH-LM on standardised residuals: LM = 21.46, p = 0.370363
    Ljung-Box on squared std residuals (lag 20): Q = 20.80, p = 0.409036
    



#### Interpreting GARCH(1,1)

Persistence is alpha + beta = 0.9792. A shock to volatility decays
with a half-life of 33.0 trading days — roughly
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
yesterday's variance estimate. Alpha (0.1195) is the reactive
component, the weight placed on yesterday's squared shock. Volatility under
this model is a slow-moving state occasionally jolted by news, not a fresh
draw each day.

The standardised residuals now return
LM = 21.46
(p = 0.3704) on the ARCH-LM test,
corroborated by the Ljung-Box test on squared residuals
(Q = 20.80,
p = 0.4090).
The variance dynamics are captured; no significant clustering remains in either test.

What the Normal-innovation fit cannot fix is the distribution itself. The
model assumes standardised residuals are Gaussian; the data said otherwise
in Notebook 02 (excess kurtosis 10.6566). That mismatch distorts the
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
assumption the data rejected in Notebook 02: excess kurtosis 10.6566,
skewness -0.3493, Jarque–Bera 31,748.52
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
    Vol Model:                            GARCH   Log-Likelihood:               -8555.85
    Distribution:      Standardized Student's t   AIC:                           17121.7
    Method:                  Maximum Likelihood   BIC:                           17155.5
                                                  No. Observations:                 6412
    Date:                      Tue, Aug 18 2026   Df Residuals:                     6411
    Time:                              23:37:17   Df Model:                            1
                                     Mean Model                                 
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    mu             0.0819  8.952e-03      9.146  5.904e-20 [6.433e-02,9.942e-02]
                                  Volatility Model                              
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    omega          0.0160  3.203e-03      4.991  5.993e-07 [9.710e-03,2.227e-02]
    alpha[1]       0.1236  1.080e-02     11.445  2.505e-30     [  0.102,  0.145]
    beta[1]        0.8707  1.038e-02     83.916      0.000     [  0.850,  0.891]
                                  Distribution                              
    ========================================================================
                     coef    std err          t      P>|t|  95.0% Conf. Int.
    ------------------------------------------------------------------------
    nu             6.0892      0.473     12.874  6.328e-38 [  5.162,  7.016]
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
      <td>-8555.852278</td>
      <td>17121.704556</td>
      <td>17155.534189</td>
      <td>5.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>GARCH(1,1) — Normal</th>
      <td>-8711.002062</td>
      <td>17430.004123</td>
      <td>17457.067829</td>
      <td>4.0</td>
      <td>308.299567</td>
      <td>-155.149783</td>
    </tr>
    <tr>
      <th>ARCH(1) — Normal</th>
      <td>-9778.957602</td>
      <td>19563.915203</td>
      <td>19584.212983</td>
      <td>3.0</td>
      <td>2442.210647</td>
      <td>-1223.105323</td>
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

    nu (degrees of freedom): 6.09
    alpha[1]: 0.1236
    beta[1]: 0.8707
    Persistence (alpha + beta): 0.9943
    Volatility half-life: 122.3 trading days
    Log-likelihood gain vs Normal: 155.15
    AIC improvement vs Normal: 308.30
    ARCH-LM on standardised residuals: LM = 23.73, p = 0.254258
    



#### Interpreting the Student's t fit

The estimated degrees of freedom is nu = 6.09.
This is deep in heavy-tail territory: after accounting for time-varying variance, daily shocks remain far from Gaussian, and the model now says so explicitly instead of distorting other parameters to compensate.

The log-likelihood improves by 155.15 and AIC falls by 308.30
from one added parameter. The variance equation is identical across the two
specifications, so the entire gain comes from the distributional assumption:
the heavy tails documented in Notebook 02 (excess kurtosis 10.6566) are
better represented by a Student's t than by a Gaussian. Persistence moves
from 0.9792 under the Normal to 0.9943 here — a
measure of how much the distributional misspecification was leaking into the
variance estimates.

On the ARCH-LM test the standardised residuals return
LM = 23.73
(p = 0.2543).
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

    Excess kurtosis, raw returns (pandas G2):        11.3102
    Excess kurtosis, standardised residuals (G2):    2.0741
    Locked NB02 value for comparison:                10.6566
    




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
falls from 11.3102 in the raw returns to 2.0741 in the
standardised residuals (both computed with the pandas bias-corrected G2
estimator, the same estimator used for the 10.6566 locked in Notebook 02;
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
disagrees with that symmetry. Skewness of -0.3493 was documented in
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
    Vol Model:                        GJR-GARCH   Log-Likelihood:               -8447.40
    Distribution:      Standardized Student's t   AIC:                           16906.8
    Method:                  Maximum Likelihood   BIC:                           16947.4
                                                  No. Observations:                 6412
    Date:                      Tue, Aug 18 2026   Df Residuals:                     6411
    Time:                              23:37:17   Df Model:                            1
                                     Mean Model                                 
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    mu             0.0540  9.041e-03      5.970  2.379e-09 [3.625e-02,7.169e-02]
                                   Volatility Model                              
    =============================================================================
                     coef    std err          t      P>|t|       95.0% Conf. Int.
    -----------------------------------------------------------------------------
    omega          0.0186  3.217e-03      5.783  7.328e-09  [1.230e-02,2.491e-02]
    alpha[1]       0.0000  9.070e-03      0.000      1.000 [-1.778e-02,1.778e-02]
    gamma[1]       0.2037  2.086e-02      9.767  1.559e-22      [  0.163,  0.245]
    beta[1]        0.8809  1.246e-02     70.676      0.000      [  0.857,  0.905]
                                  Distribution                              
    ========================================================================
                     coef    std err          t      P>|t|  95.0% Conf. Int.
    ------------------------------------------------------------------------
    nu             6.7019      0.585     11.451  2.317e-30 [  5.555,  7.849]
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
      <td>-8447.397385</td>
      <td>16906.794771</td>
      <td>16947.390330</td>
      <td>6.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>GARCH(1,1) — Student's t</th>
      <td>-8555.852278</td>
      <td>17121.704556</td>
      <td>17155.534189</td>
      <td>5.0</td>
      <td>214.909785</td>
      <td>-108.454893</td>
    </tr>
    <tr>
      <th>GARCH(1,1) — Normal</th>
      <td>-8711.002062</td>
      <td>17430.004123</td>
      <td>17457.067829</td>
      <td>4.0</td>
      <td>523.209352</td>
      <td>-263.604676</td>
    </tr>
    <tr>
      <th>ARCH(1) — Normal</th>
      <td>-9778.957602</td>
      <td>19563.915203</td>
      <td>19584.212983</td>
      <td>3.0</td>
      <td>2657.120432</td>
      <td>-1331.560216</td>
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
    gamma[1]: 0.2037 (p = 1.56e-22)
    beta[1]: 0.8809
    Persistence (alpha + gamma/2 + beta): 0.9828
    AIC change vs symmetric GARCH-t: +214.91
    ARCH-LM on standardised residuals: LM = 18.34, p = 0.565307
    Ljung-Box on squared std residuals (lag 20): Q = 17.05, p = 0.649561
    



#### Interpreting GJR-GARCH

The asymmetry parameter is gamma = 0.2037
(p < 0.001).
The leverage effect is confirmed at the model level. A negative shock feeds into next-day variance with weight alpha + gamma = 0.2037, against 0.0000e+00 for a positive shock of the same size. The sign of a move carries information the squared shock alone discards, consistent with the negative skewness (-0.3493) documented in Notebook 02.

AIC improves by 214.91 over the symmetric Student t fit.
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
Q = 17.05
(p = 0.6496),
confirming no residual serial dependence in variance.



### 5.6 In-sample comparison and model selection


```python
show_model_table()

comparison = pd.DataFrame(model_results).T.sort_values('aic')
best_label = comparison.index[0]
best_spec = model_specs[best_label]
best_res = model_objects[best_label]

print(f'Selected specification: {best_label}')

display(Markdown(f"""
The table ranks all four fits by AIC; **{best_label}** leads and advances to
forecast evaluation, the regime classifier, and the risk summary. Two caveats
keep this honest. These are in-sample statistics computed on the same data
the models were fitted to, and the AIC comparison is valid only because every
model was fitted on the same scaled series, so the rescaling constant cancels.
The delta columns restate the ranking in relative terms: delta_aic is each
model's AIC distance from the leader, and by the Burnham–Anderson guideline a
distance above 10 means essentially no support for the trailing model.

In-sample fit does not establish forecasting ability. That question belongs
to the next section, which evaluates the winning specification against the
persistence benchmark on data neither has seen.
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
      <th>GJR-GARCH(1,1,1) — Student's t</th>
      <td>-8447.397385</td>
      <td>16906.794771</td>
      <td>16947.390330</td>
      <td>6.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>GARCH(1,1) — Student's t</th>
      <td>-8555.852278</td>
      <td>17121.704556</td>
      <td>17155.534189</td>
      <td>5.0</td>
      <td>214.909785</td>
      <td>-108.454893</td>
    </tr>
    <tr>
      <th>GARCH(1,1) — Normal</th>
      <td>-8711.002062</td>
      <td>17430.004123</td>
      <td>17457.067829</td>
      <td>4.0</td>
      <td>523.209352</td>
      <td>-263.604676</td>
    </tr>
    <tr>
      <th>ARCH(1) — Normal</th>
      <td>-9778.957602</td>
      <td>19563.915203</td>
      <td>19584.212983</td>
      <td>3.0</td>
      <td>2657.120432</td>
      <td>-1331.560216</td>
    </tr>
  </tbody>
</table>
</div>


    Selected specification: GJR-GARCH(1,1,1) — Student's t
    



The table ranks all four fits by AIC; **GJR-GARCH(1,1,1) — Student's t** leads and advances to
forecast evaluation, the regime classifier, and the risk summary. Two caveats
keep this honest. These are in-sample statistics computed on the same data
the models were fitted to, and the AIC comparison is valid only because every
model was fitted on the same scaled series, so the rescaling constant cancels.
The delta columns restate the ranking in relative terms: delta_aic is each
model's AIC distance from the leader, and by the Burnham–Anderson guideline a
distance above 10 means essentially no support for the trailing model.

In-sample fit does not establish forecasting ability. That question belongs
to the next section, which evaluates the winning specification against the
persistence benchmark on data neither has seen.



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
  (GARCH(1,1) — Student's t), both evaluated against persistence and EWMA.
  Running the runner-up confirms the in-sample ranking holds out of sample.

One conversion is required for a fair comparison. GARCH-family models
forecast the conditional standard deviation, sigma. The persistence benchmark
forecasts the absolute return directly. These are different quantities: the
expected absolute value of a shock is c × sigma, where c depends on the
innovation distribution — sqrt(2/pi) ≈ 0.7979 for a Normal, and roughly 0.75
to 0.80 for a standardised Student's t at the nu values typical of equity
indices. Comparing raw sigma against realised |r| would penalise the model
for a unit mismatch rather than forecasting skill. GARCH forecasts below are
converted to expected absolute returns using the exact factor implied by each
refit's fitted distribution, and the unadjusted sigma error is reported
alongside to show the size of the effect. EWMA forecasts are already in
absolute-return units (sqrt of variance), so no distributional conversion is
needed.
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
- Forecasters: the AIC winner (GJR-GARCH(1,1,1) — Student's t) and the runner-up
  (GARCH(1,1) — Student's t), both evaluated against persistence and EWMA.
  Running the runner-up confirms the in-sample ranking holds out of sample.

One conversion is required for a fair comparison. GARCH-family models
forecast the conditional standard deviation, sigma. The persistence benchmark
forecasts the absolute return directly. These are different quantities: the
expected absolute value of a shock is c × sigma, where c depends on the
innovation distribution — sqrt(2/pi) ≈ 0.7979 for a Normal, and roughly 0.75
to 0.80 for a standardised Student's t at the nu values typical of equity
indices. Comparing raw sigma against realised |r| would penalise the model
for a unit mismatch rather than forecasting skill. GARCH forecasts below are
converted to expected absolute returns using the exact factor implied by each
refit's fitted distribution, and the unadjusted sigma error is reported
alongside to show the size of the effect. EWMA forecasts are already in
absolute-return units (sqrt of variance), so no distributional conversion is
needed.




```python

def abs_return_factor(params):
    """E|Z| for the fitted innovation distribution.

    Student's t (standardised, nu degrees of freedom) when 'nu' is present;
    Normal otherwise.
    """
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
runnerup_label = "GARCH(1,1) — Student's t"
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

# EWMA benchmark on the test window (expanding from the same history)
ewma_var_full = (returns ** 2).ewm(alpha=(1 - ewma_lambda), adjust=False).mean()
wf['ewma_forecast'] = np.sqrt(ewma_var_full.shift(1)).reindex(wf.index)

wf = wf.dropna()

print(f'Winner:    {best_label}')
print(f'Runner-up: {runnerup_label}')
print(f'Walk-forward window: {wf.index.min().date()} to {wf.index.max().date()}')
print(f'Forecasts produced: {len(wf)}')
```

    Winner:    GJR-GARCH(1,1,1) — Student's t
    Runner-up: GARCH(1,1) — Student's t
    Walk-forward window: 2025-08-14 to 2026-08-14
    Forecasts produced: 252
    


```python
# ── Error metrics ──
def qlike(actual, forecast):
    """QLIKE loss: mean of (actual^2 / forecast^2 - log(actual^2 / forecast^2) - 1).
    Patton (2011) consistent loss for volatility proxy evaluation."""
    ratio = (actual ** 2) / (forecast ** 2)
    return np.mean(ratio - np.log(ratio) - 1)

rmse_garch_wf = root_mean_squared_error(wf['realised_vol'], wf['abs_return_forecast'])
mae_garch_wf = mean_absolute_error(wf['realised_vol'], wf['abs_return_forecast'])
qlike_garch_wf = qlike(wf['realised_vol'], wf['abs_return_forecast'])
rmse_sigma_wf = root_mean_squared_error(wf['realised_vol'], wf['sigma_forecast'])

rmse_ru_wf = root_mean_squared_error(wf['realised_vol'], wf['abs_return_forecast_ru'])
mae_ru_wf = mean_absolute_error(wf['realised_vol'], wf['abs_return_forecast_ru'])
qlike_ru_wf = qlike(wf['realised_vol'], wf['abs_return_forecast_ru'])

rmse_pers_wf = root_mean_squared_error(wf['realised_vol'], wf['persistence_forecast'])
mae_pers_wf = mean_absolute_error(wf['realised_vol'], wf['persistence_forecast'])
qlike_pers_wf = np.nan  # QLIKE divides by forecast², undefined when persistence ≈ 0

rmse_ewma_wf = root_mean_squared_error(wf['realised_vol'], wf['ewma_forecast'])
mae_ewma_wf = mean_absolute_error(wf['realised_vol'], wf['ewma_forecast'])
qlike_ewma_wf = qlike(wf['realised_vol'], wf['ewma_forecast'])

improvement_vs_pers = (rmse_pers_wf - rmse_garch_wf) / rmse_pers_wf * 100
improvement_vs_ewma = (rmse_ewma_wf - rmse_garch_wf) / rmse_ewma_wf * 100

forecast_comparison = pd.DataFrame({
    'RMSE': [rmse_pers_wf, rmse_ewma_wf, rmse_garch_wf, rmse_ru_wf, rmse_sigma_wf],
    'MAE':  [mae_pers_wf, mae_ewma_wf, mae_garch_wf, mae_ru_wf, np.nan],
    'QLIKE': [qlike_pers_wf, qlike_ewma_wf, qlike_garch_wf, qlike_ru_wf, np.nan],
}, index=[
    'Persistence (test window)',
    f'EWMA λ={ewma_lambda} (test window)',
    f'{best_label} (E|r| adjusted)',
    f'{runnerup_label} (E|r| adjusted)',
    f'{best_label} (raw sigma)',
])
display(forecast_comparison)

print(f'RMSE change vs persistence: {improvement_vs_pers:+.1f}%')
print(f'RMSE change vs EWMA:        {improvement_vs_ewma:+.1f}%')

# ── Diebold-Mariano test: GJR-GARCH vs persistence ──
# Squared-error loss differential, Newey-West HAC variance for serial correlation.
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

# ── Diebold-Mariano under QLIKE loss: EWMA vs GJR-GARCH ──
# EWMA wins on QLIKE in the table. Is that reversal significant?
def qlike_loss(actual, forecast):
    """Per-observation QLIKE loss (not averaged)."""
    ratio = (actual ** 2) / (forecast ** 2)
    return ratio - np.log(ratio) - 1

ql_garch = qlike_loss(wf['realised_vol'].values, wf['abs_return_forecast'].values)
ql_ewma = qlike_loss(wf['realised_vol'].values, wf['ewma_forecast'].values)
d_qlike_ewma = ql_ewma - ql_garch  # positive = GARCH better under QLIKE

dm_stat_qlike_ewma, dm_p_qlike_ewma = dm_test_nw(d_qlike_ewma)


print(f'\nDiebold-Mariano (Newey-West HAC):')
print(f'  vs Persistence (SE loss): DM = {dm_stat_pers:.3f}, p = {dm_p_pers:.4f}')
print(f'  vs EWMA (SE loss):        DM = {dm_stat_ewma:.3f}, p = {dm_p_ewma:.4f}')
print(f'  vs EWMA (QLIKE loss):     DM = {dm_stat_qlike_ewma:.3f}, p = {dm_p_qlike_ewma:.4f}')

display(Markdown(f"""
### Walk-forward results

Both benchmarks and both GARCH specifications were evaluated on the same
{len(wf)}-day test window. QLIKE (Patton 2011) is reported alongside RMSE
and MAE because it remains a consistent loss function even when the
volatility proxy is noisy — a property squared-error loss does not have.

**Persistence** produced RMSE = {rmse_pers_wf:.6f}.
**EWMA (λ = {ewma_lambda})** produced RMSE = {rmse_ewma_wf:.6f} — already
{"a large improvement over persistence, confirming that most of the persistence benchmark's weakness is its reliance on a single noisy observation rather than smooth decay." if rmse_ewma_wf < rmse_pers_wf * 0.9 else "a moderate improvement over persistence."}
**{best_label}** (E|r|-adjusted) produced
RMSE = {rmse_garch_wf:.6f}, a {improvement_vs_pers:.1f}% reduction vs
persistence and a {improvement_vs_ewma:+.1f}% change vs EWMA.
{'The model beats both benchmarks, including the tougher EWMA bar. The gain over EWMA is the value of estimated (rather than fixed) decay and distributional modelling.' if improvement_vs_ewma > 0 else 'The model beats persistence but not EWMA on RMSE. Against EWMA the edge is negative, meaning EWMA already captures most of the forecastable structure. The estimated GARCH parameters add distributional modelling and asymmetry but do not translate into lower point-forecast error over this particular window.'}

**{runnerup_label}** produced RMSE = {rmse_ru_wf:.6f}.
{'The in-sample AIC ranking holds out of sample: the asymmetric specification retains its edge.' if rmse_garch_wf < rmse_ru_wf else 'The runner-up matches or beats the winner on RMSE, suggesting the in-sample AIC advantage from asymmetry does not translate into superior point forecasts over this window. Both remain valid for the regime classifier, and the AIC-selected specification is retained for consistency.'}

The Diebold-Mariano test (Newey-West HAC variance, squared-error loss) asks
whether each RMSE difference is distinguishable from noise. Against
persistence, DM = {dm_stat_pers:.3f}
(p {'< 0.001' if dm_p_pers < 0.001 else f'= {dm_p_pers:.4f}'}) —
{'a statistically significant improvement.' if dm_p_pers < 0.05 else 'not significant at the 5% level. The 252-day window is one draw, and the point estimate should not be over-interpreted.'}
Against EWMA, DM = {dm_stat_ewma:.3f}
(p {'< 0.001' if dm_p_ewma < 0.001 else f'= {dm_p_ewma:.4f}'}) —
{'a significant edge over the industry-standard naive forecast.' if dm_p_ewma < 0.05 else 'not significant at the 5% level against the tougher benchmark.'}

The QLIKE column tells a different story. EWMA scores lower (better) than
both GARCH specifications. QLIKE penalises proportional forecast errors more
heavily than squared-error loss, so it is more sensitive to days where the
model overshoots relative to realised volatility. A Diebold-Mariano test
under QLIKE loss returns DM = {dm_stat_qlike_ewma:.3f}
(p = {dm_p_qlike_ewma:.4f}) —
{'the EWMA advantage under QLIKE is statistically significant, a genuine disagreement between loss functions rather than noise.' if dm_p_qlike_ewma < 0.05 else 'not significant at the 5% level. The EWMA advantage under QLIKE is directional but not distinguishable from noise in this window.'}
Persistence is excluded from QLIKE because its near-zero forecasts on flat
days make the loss undefined.

For reference, the unadjusted sigma forecasts score
RMSE = {rmse_sigma_wf:.6f} against the same target. The difference between
the two {best_label} rows is the unit-mismatch effect the conversion removes,
not a change in forecasting skill.
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
      <td>0.007438</td>
      <td>0.005514</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>EWMA λ=0.94 (test window)</th>
      <td>0.005683</td>
      <td>0.004586</td>
      <td>1.680345</td>
    </tr>
    <tr>
      <th>GJR-GARCH(1,1,1) — Student's t (E|r| adjusted)</th>
      <td>0.005206</td>
      <td>0.003964</td>
      <td>1.825429</td>
    </tr>
    <tr>
      <th>GARCH(1,1) — Student's t (E|r| adjusted)</th>
      <td>0.005332</td>
      <td>0.004084</td>
      <td>1.881063</td>
    </tr>
    <tr>
      <th>GJR-GARCH(1,1,1) — Student's t (raw sigma)</th>
      <td>0.005714</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    RMSE change vs persistence: +30.0%
    RMSE change vs EWMA:        +8.4%
    
    Diebold-Mariano (Newey-West HAC):
      vs Persistence (SE loss): DM = 5.246, p = 0.0000
      vs EWMA (SE loss):        DM = 4.261, p = 0.0000
      vs EWMA (QLIKE loss):     DM = -1.517, p = 0.1293
    



### Walk-forward results

Both benchmarks and both GARCH specifications were evaluated on the same
252-day test window. QLIKE (Patton 2011) is reported alongside RMSE
and MAE because it remains a consistent loss function even when the
volatility proxy is noisy — a property squared-error loss does not have.

**Persistence** produced RMSE = 0.007438.
**EWMA (λ = 0.94)** produced RMSE = 0.005683 — already
a large improvement over persistence, confirming that most of the persistence benchmark's weakness is its reliance on a single noisy observation rather than smooth decay.
**GJR-GARCH(1,1,1) — Student's t** (E|r|-adjusted) produced
RMSE = 0.005206, a 30.0% reduction vs
persistence and a +8.4% change vs EWMA.
The model beats both benchmarks, including the tougher EWMA bar. The gain over EWMA is the value of estimated (rather than fixed) decay and distributional modelling.

**GARCH(1,1) — Student's t** produced RMSE = 0.005332.
The in-sample AIC ranking holds out of sample: the asymmetric specification retains its edge.

The Diebold-Mariano test (Newey-West HAC variance, squared-error loss) asks
whether each RMSE difference is distinguishable from noise. Against
persistence, DM = 5.246
(p < 0.001) —
a statistically significant improvement.
Against EWMA, DM = 4.261
(p < 0.001) —
a significant edge over the industry-standard naive forecast.

The QLIKE column tells a different story. EWMA scores lower (better) than
both GARCH specifications. QLIKE penalises proportional forecast errors more
heavily than squared-error loss, so it is more sensitive to days where the
model overshoots relative to realised volatility. A Diebold-Mariano test
under QLIKE loss returns DM = -1.517
(p = 0.1293) —
not significant at the 5% level. The EWMA advantage under QLIKE is directional but not distinguishable from noise in this window.
Persistence is excluded from QLIKE because its near-zero forecasts on flat
days make the loss undefined.

For reference, the unadjusted sigma forecasts score
RMSE = 0.005714 against the same target. The difference between
the two GJR-GARCH(1,1,1) — Student's t rows is the unit-mismatch effect the conversion removes,
not a change in forecasting skill.




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
selected model (GJR-GARCH(1,1,1) — Student's t), using percentile thresholds:

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
      Calm   < 0.1024
      Normal < 0.1941
      Stress < 0.3380
      Crisis >= 0.3380
    
    regime
    Normal    3206
    Calm      1603
    Stress    1282
    Crisis     321
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
    Stress    0.582
    Normal    0.319
    Crisis    0.096
    Calm      0.004
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



### Risk intelligence summary — as of 2026-08-14

| Quantity | Value |
|---|---|
| Next-day conditional standard deviation | 0.6317% daily (10.03% annualised) |
| Expected absolute daily move | 0.4781% |
| Historical percentile | 23% |
| Regime | **Calm** |

Volatility sits in the bottom quartile of its historical distribution. The forecast is produced by GJR-GARCH(1,1,1) — Student's t,
fitted on the full sample, and the regime label applies the thresholds
validated in section 7.

The two forecast rows are different units of the same forecast, and the
distinction matters for anyone translating these numbers into potential
losses. The conditional standard deviation is the model's sigma: the input to
VaR-style calculations, where a loss threshold is sigma scaled by a quantile
of the fitted innovation distribution. The expected absolute daily move is
c × sigma, the expectation of tomorrow's |return| — a typical move, not a
bad one. Turning the 10.0% annualised figure into a potential
daily loss uses the standard deviation row and a chosen confidence level,
never the absolute-move row.



## 9. What this notebook established


```python
display(Markdown(f"""
The ARCH-LM test (LM = {arch_test[0]:,.2f}, p ≈ 0) confirmed conditional
heteroskedasticity in daily returns. ARCH(1) validated the mechanism but not
the memory: significant ARCH effects survived in its standardised residuals
(both ARCH-LM and Ljung-Box on squared residuals). GARCH(1,1) fixed that
with one parameter. The distributional change from Normal to Student's t
(nu = {nu:.2f}) improved AIC by {aic_drop:,.2f} with the variance equation
held fixed, isolating the value of modelling the tails honestly.
{f"GJR-GARCH then confirmed the leverage effect: gamma = {gamma_j:.4f}, so negative shocks raise next-day variance with weight {alpha_j + gamma_j:.4f} against {alpha_j:.4e} for positive shocks of the same size. Alpha pinning at its lower bound means all shock-driven variance flows through the asymmetry channel, a boundary result flagged in section 5.5." if gamma_p < 0.05 and gamma_j > 0 else f"The GJR asymmetry term was tested and found non-significant: the unconditional skewness of {skew_nb02:.4f} does not translate into a conditional-variance asymmetry in this sample, a null result reported as found."}
The AIC winner, {best_label}, advanced to every downstream section.

Out of sample, two benchmarks were tested rather than one: persistence
(single-lag absolute return) and EWMA (RiskMetrics λ = {ewma_lambda}), the
industry-standard naive forecast. {best_label}
{f"beat persistence by {improvement_vs_pers:.1f}% on RMSE (DM p {'< 0.001' if dm_p_pers < 0.001 else f'= {dm_p_pers:.4f}'}) and {'also beat EWMA by ' + f'{improvement_vs_ewma:.1f}%' + ' (DM p ' + ('<0.001' if dm_p_ewma < 0.001 else f'= {dm_p_ewma:.4f}') + ')' if improvement_vs_ewma > 0 else 'did not beat EWMA on RMSE, meaning EWMA already captures most forecastable structure in this window'} over the final 252 trading days" if improvement_vs_pers > 0 else f"did not beat the persistence benchmark over the final 252 trading days ({improvement_vs_pers:+.1f}% RMSE change), a null result reported as found"}.
The runner-up ({runnerup_label}) was also evaluated out of sample
({'confirming the in-sample AIC ranking holds' if rmse_garch_wf < rmse_ru_wf else 'with no clear OOS advantage for the asymmetric specification, though it is retained as the AIC-selected model'}).
QLIKE (Patton 2011) was reported alongside RMSE and MAE as a
proxy-consistent loss function. Under QLIKE, EWMA scored lower than both
GARCH specifications — a directional reversal of the RMSE ranking, though
not significant at the 5% level (DM = {dm_stat_qlike_ewma:.3f},
p = {dm_p_qlike_ewma:.4f}).

The regime classifier's validation against the 2020 COVID crash and the 2022
rate-hike cycle is reported in section 7, and the notebook closes with the
daily risk intelligence summary the Daily Market Risk Report will consume —
stated in both of its units, conditional standard deviation and expected
absolute move.

### Limitations

This evaluation uses a single test window of 252 trading days. That mirrors
the Notebook 04 convention and keeps every model on identical footing, but
one window is one draw from one market regime, and forecasting performance
can differ under others. The Diebold-Mariano tests quantify whether the
observed differences are distinguishable from noise within this window, but
they do not guarantee the ranking generalises. An extended study would repeat
the walk-forward over multiple rolling or expanding windows and report the
distribution of outcomes. The regime thresholds share a related caveat, noted
in section 7: they are computed on the full sample for descriptive
validation, where a production system would freeze them on a training window.

### Evaluation protocol for Notebook 06

Notebook 06 inherits this evaluation protocol unchanged: the same
realised-volatility target (absolute daily log returns), the same 252-day
walk-forward window, the same 21-day refit schedule, the same persistence
and EWMA benchmarks, and the same RMSE, MAE, QLIKE, and Diebold-Mariano
metrics. Any deep learning result reported there is directly comparable to
the figures above, so a difference between the two reflects the models, not
the test.
"""))
```



The ARCH-LM test (LM = 1,857.96, p ≈ 0) confirmed conditional
heteroskedasticity in daily returns. ARCH(1) validated the mechanism but not
the memory: significant ARCH effects survived in its standardised residuals
(both ARCH-LM and Ljung-Box on squared residuals). GARCH(1,1) fixed that
with one parameter. The distributional change from Normal to Student's t
(nu = 6.09) improved AIC by 308.30 with the variance equation
held fixed, isolating the value of modelling the tails honestly.
GJR-GARCH then confirmed the leverage effect: gamma = 0.2037, so negative shocks raise next-day variance with weight 0.2037 against 0.0000e+00 for positive shocks of the same size. Alpha pinning at its lower bound means all shock-driven variance flows through the asymmetry channel, a boundary result flagged in section 5.5.
The AIC winner, GJR-GARCH(1,1,1) — Student's t, advanced to every downstream section.

Out of sample, two benchmarks were tested rather than one: persistence
(single-lag absolute return) and EWMA (RiskMetrics λ = 0.94), the
industry-standard naive forecast. GJR-GARCH(1,1,1) — Student's t
beat persistence by 30.0% on RMSE (DM p < 0.001) and also beat EWMA by 8.4% (DM p <0.001) over the final 252 trading days.
The runner-up (GARCH(1,1) — Student's t) was also evaluated out of sample
(confirming the in-sample AIC ranking holds).
QLIKE (Patton 2011) was reported alongside RMSE and MAE as a
proxy-consistent loss function. Under QLIKE, EWMA scored lower than both
GARCH specifications — a directional reversal of the RMSE ranking, though
not significant at the 5% level (DM = -1.517,
p = 0.1293).

The regime classifier's validation against the 2020 COVID crash and the 2022
rate-hike cycle is reported in section 7, and the notebook closes with the
daily risk intelligence summary the Daily Market Risk Report will consume —
stated in both of its units, conditional standard deviation and expected
absolute move.

### Limitations

This evaluation uses a single test window of 252 trading days. That mirrors
the Notebook 04 convention and keeps every model on identical footing, but
one window is one draw from one market regime, and forecasting performance
can differ under others. The Diebold-Mariano tests quantify whether the
observed differences are distinguishable from noise within this window, but
they do not guarantee the ranking generalises. An extended study would repeat
the walk-forward over multiple rolling or expanding windows and report the
distribution of outcomes. The regime thresholds share a related caveat, noted
in section 7: they are computed on the full sample for descriptive
validation, where a production system would freeze them on a training window.

### Evaluation protocol for Notebook 06

Notebook 06 inherits this evaluation protocol unchanged: the same
realised-volatility target (absolute daily log returns), the same 252-day
walk-forward window, the same 21-day refit schedule, the same persistence
and EWMA benchmarks, and the same RMSE, MAE, QLIKE, and Diebold-Mariano
metrics. Any deep learning result reported there is directly comparable to
the figures above, so a difference between the two reflects the models, not
the test.




```python
for name, obj in list(globals().items()):
    if type(obj).__name__ == 'ARCHModelResult':
        print(name)
```

    res_arch1
    res_garch11
    res_garch11_t
    res_gjr
    best_res
    res_refit
    res_refit_ru
    


```python
# ── Export metrics for downstream notebooks ──
metrics_path = Path('../data/locked_metrics.json')
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

metrics['notebook_05'] = {
    'best_label':           best_label,
    'wf_rmse_garch':        float(rmse_garch_wf),
    'wf_mae_garch':         float(mae_garch_wf),
    'wf_rmse_persistence':  float(rmse_pers_wf),
    'wf_mae_persistence':   float(mae_pers_wf),
    'wf_rmse_ewma':         float(rmse_ewma_wf),
    'wf_mae_ewma':          float(mae_ewma_wf),
    'improvement_rmse_pct': float(improvement_vs_pers),
    'improvement_vs_ewma_pct': float(improvement_vs_ewma),
    'dm_stat_vs_persistence': float(dm_stat_pers),
    'dm_p_vs_persistence':    float(dm_p_pers),
    'dm_stat_vs_ewma':        float(dm_stat_ewma),
    'dm_p_vs_ewma':           float(dm_p_ewma),
    'dm_stat_vs_ewma_qlike':  float(dm_stat_qlike_ewma),
    'dm_p_vs_ewma_qlike':     float(dm_p_qlike_ewma),
    'ewma_lambda':            float(ewma_lambda),
}

try:
    p = best_res.params
    a = sum(v for k, v in p.items() if k.startswith('alpha'))
    b = sum(v for k, v in p.items() if k.startswith('beta'))
    g = sum(v for k, v in p.items() if k.startswith('gamma'))
    metrics['notebook_05']['garch_persistence'] = float(a + b + g / 2)
except (NameError, AttributeError, KeyError) as e:
    print(f'Skipped garch_persistence: {e}')

metrics_path.write_text(json.dumps(metrics, indent=2))

print(f'Exported notebook_05 metrics to {metrics_path.resolve()}')
for k, v in metrics['notebook_05'].items():
    print(f'  {k}: {v}')
```

    Exported notebook_05 metrics to C:\Users\Mena\Documents\Python\sp500-market-intelligence\data\locked_metrics.json
      best_label: GJR-GARCH(1,1,1) — Student's t
      wf_rmse_garch: 0.005206427586318169
      wf_mae_garch: 0.003964307292970376
      wf_rmse_persistence: 0.007438077359303796
      wf_mae_persistence: 0.0055138317645078054
      wf_rmse_ewma: 0.0056829434700153775
      wf_mae_ewma: 0.0045862660107591074
      improvement_rmse_pct: 30.00304601825907
      improvement_vs_ewma_pct: 8.385018894019003
      dm_stat_vs_persistence: 5.246419922106737
      dm_p_vs_persistence: 1.5508320849733082e-07
      dm_stat_vs_ewma: 4.26138434011704
      dm_p_vs_ewma: 2.0316449053758845e-05
      dm_stat_vs_ewma_qlike: -1.5170117493943702
      dm_p_vs_ewma_qlike: 0.12926371760874122
      ewma_lambda: 0.94
      garch_persistence: 0.9828090407814376
    


```python

```
