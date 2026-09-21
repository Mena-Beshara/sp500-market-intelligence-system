# Classical Time Series Forecasting for S&P 500

This notebook establishes forecasting benchmarks using classical statistical
models, building on the diagnostics from Notebook 02.

Daily S&P 500 log returns are stationary, show fat tails and negative skewness,
exhibit weak directional autocorrelation, and show strong volatility clustering
with conditional heteroskedasticity. These properties suggest the conditional
mean may be hard to forecast, while volatility carries more structure.

## Objectives

- Establish baseline performance using simple forecasting methods.
- Evaluate classical models with statistically justified parameter selection
  and out-of-sample testing.
- Test Prophet as a seasonal benchmark.
- Run residual diagnostics on the fitted models.
- Determine whether classical models extract predictive signal beyond the
  baselines.


```python
# Import the needed libraries
import pandas as pd 
import numpy as np 
import plotly.io as pio
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import json
from pathlib import Path
import warnings 
warnings.filterwarnings('ignore')
# Setting Plotly backend plotting for pandas 
pd.options.plotting.backend = 'plotly'
pio.templates.default = 'plotly_dark'
import plotly.express as px 
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
from IPython.display import display, Markdown
import tabulate

```

# Load Dataset


```python
# Load dataset with features 
df = pd.read_parquet('../data/sp500_features.parquet')

# Ensure the Date Order 
df = df.sort_index()

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
# Dataset Structure
print(f'Rows: {len(df):,}')
print(f'Columns: {df.shape[1]}')
print(f'Date Range: {df.index.min()} -> {df.index.max()}')
```

    Rows: 6,412
    Columns: 64
    Date Range: 2001-02-13 00:00:00 -> 2026-08-14 00:00:00
    


```python
display(Markdown(f"""
## Load dataset with features

This notebook uses the feature-engineered dataset from Notebook 03, containing
lagged features, rolling statistics, technical indicators, calendar effects,
and volatility regime features derived from daily log returns.

Loaded {len(df):,} rows spanning {df.index.min().date()} to {df.index.max().date()}.

A chronological train-test split is established first to simulate a realistic
forecasting environment.
"""))
```



## Load dataset with features

This notebook uses the feature-engineered dataset from Notebook 03, containing
lagged features, rolling statistics, technical indicators, calendar effects,
and volatility regime features derived from daily log returns.

Loaded 6,412 rows spanning 2001-02-13 to 2026-08-14.

A chronological train-test split is established first to simulate a realistic
forecasting environment.



# Forecast Target


```python
target = 'log_returns'

y = df[target]
```

## Forecast target

The target variable is `log_returns`, the daily S&P 500 log return. Log
returns are additive through time and stationary, making them more suitable
for time-series modelling than raw prices.

The objective is to forecast next-day log returns using historical market
information.

# Train Test Split


```python
# We will use 80/20 split 
train_size = int(len(df) * 0.8)

train = df.iloc[:train_size]
test = df.iloc[train_size:]

print(f'Train observations: {len(train):,}')
print(f'Test observations: {len(test):,}')

print(f'\nTrain Period:')
print(f'{train.index.min()} -> {train.index.max()}')

print(f'Test Period:')
print(f'{test.index.min()} -> {test.index.max()}')
```

    Train observations: 5,129
    Test observations: 1,283
    
    Train Period:
    2001-02-13 00:00:00 -> 2021-07-02 00:00:00
    Test Period:
    2021-07-06 00:00:00 -> 2026-08-14 00:00:00
    


```python
display(Markdown(f"""
## Train-test split

An 80/20 chronological split produces {len(train):,} training observations
({train.index.min().date()} to {train.index.max().date()}) and {len(test):,}
test observations ({test.index.min().date()} to {test.index.max().date()}).
"""))
```



## Train-test split

An 80/20 chronological split produces 5,129 training observations
(2001-02-13 to 2021-07-02) and 1,283
test observations (2021-07-06 to 2026-08-14).




```python
fig = y.plot(
    title='Train-Test Split for Forecasting'
)

fig.add_vline(
    x=test.index[0],
    line_dash='dash'
)
fig.add_vline(x=str(test.index[0]), line_dash='dash')

fig.show()
```



# Baseline Forecasts


```python
# BENCHMARK MODELS 
target = 'log_returns'

# 1. Historical Mean
mean_forecast = np.full(len(test), train[target].mean())

# 2. Naive (Random Walk)
naive_forecast = test['return_lag_1']

# 3. Zero Return (predict no change)
zero_forecast = pd.Series(0.0, index=test.index)

```


```python
# Evaluation Function
def evaluate_forecast(actual, forecast, model_name):

    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    mae = mean_absolute_error(actual, forecast)
    rmse = root_mean_squared_error(actual, forecast)
    directional_acc = np.mean(np.sign(actual) == np.sign(forecast))

    return pd.DataFrame({
        'Model' : [model_name],
        'MAE' : [mae], 
        'RMSE' : [rmse],
        'Directional Accuracy' : [directional_acc]
    })


```


```python
benchmark_results = pd.concat([
    evaluate_forecast(test[target], mean_forecast, 'Historical Mean'),
    evaluate_forecast(test[target], naive_forecast, 'Naive Momentum'),
    evaluate_forecast(test[target], zero_forecast, 'Zero Return'),
      ], 
      ignore_index=True)
```


```python
# View Benchmark Results
benchmark_results.sort_values('RMSE')
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
      <th>Model</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>Directional Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Historical Mean</td>
      <td>0.007573</td>
      <td>0.010642</td>
      <td>0.537023</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Zero Return</td>
      <td>0.007586</td>
      <td>0.010649</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Naive Momentum</td>
      <td>0.010987</td>
      <td>0.015177</td>
      <td>0.500390</td>
    </tr>
  </tbody>
</table>
</div>




```python
hist_mean_rmse = benchmark_results.loc[benchmark_results['Model'] == 'Historical Mean', 'RMSE'].iloc[0]
hist_mean_mae = benchmark_results.loc[benchmark_results['Model'] == 'Historical Mean', 'MAE'].iloc[0]
zero_rmse = benchmark_results.loc[benchmark_results['Model'] == 'Zero Return', 'RMSE'].iloc[0]
naive_rmse = benchmark_results.loc[benchmark_results['Model'] == 'Naive Momentum', 'RMSE'].iloc[0]
```


```python
# Plot Benchmark forecasts
comparison = pd.DataFrame({
    'Actual' : test[target],
    'Historical Mean' : mean_forecast,
    'Naive Momentum' : naive_forecast,
    'Zero Return' : zero_forecast
})

comparison.iloc[:250].plot(
    title = 'Benchmark Forecast Comparison'
)
```




```python
display(Markdown(f"""
## Benchmark model results

Three benchmarks are established before fitting statistical models: the
historical mean forecast (RMSE = {hist_mean_rmse:.6f}), a naive momentum
forecast using `return_lag_1` (RMSE = {naive_rmse:.6f}), and a zero-return
forecast (RMSE = {zero_rmse:.6f}).

Given the weak autocorrelation found during diagnostics, these simple
approaches may be hard to beat.
"""))
```



## Benchmark model results

Three benchmarks are established before fitting statistical models: the
historical mean forecast (RMSE = 0.010642), a naive momentum
forecast using `return_lag_1` (RMSE = 0.015177), and a zero-return
forecast (RMSE = 0.010649).

Given the weak autocorrelation found during diagnostics, these simple
approaches may be hard to beat.



# ARIMA Model


```python
# We plot ACF and PACF for ARIMA order selection 
plot_acf(train[target], lags = 40)
plot_pacf(train[target], lags = 40 );
```


    
![png](04_Baseline_Forecasting_Models_files/04_Baseline_Forecasting_Models_22_0.png)
    



    
![png](04_Baseline_Forecasting_Models_files/04_Baseline_Forecasting_Models_22_1.png)
    


## ARIMA order selection

ARIMA requires three parameters: the autoregressive order (p), differencing
order (d), and moving average order (q).

The ACF and PACF confirm the earlier diagnostics — most autocorrelations sit
close to zero, indicating limited linear dependence in daily log returns.
Given the weak autocorrelation and confirmed stationarity, ARIMA(1,0,1) is
used as an initial specification.


```python
# Fit ARIMA Static Model 
model = ARIMA(
    train[target],
    order = (1, 0, 1)
)
arima_fit = model.fit()

print(arima_fit.summary())
```

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:            log_returns   No. Observations:                 5129
    Model:                 ARIMA(1, 0, 1)   Log Likelihood               15282.048
    Date:                Mon, 17 Aug 2026   AIC                         -30556.095
    Time:                        23:24:28   BIC                         -30529.924
    Sample:                             0   HQIC                        -30546.935
                                   - 5129                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.0002      0.000      1.423      0.155    -8.7e-05       0.001
    ar.L1         -0.0521      0.045     -1.157      0.247      -0.140       0.036
    ma.L1         -0.0680      0.046     -1.474      0.140      -0.158       0.022
    sigma2         0.0002   1.23e-06    123.094      0.000       0.000       0.000
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):             26600.03
    Prob(Q):                              1.00   Prob(JB):                         0.00
    Heteroskedasticity (H):               1.13   Skew:                            -0.58
    Prob(H) (two-sided):                  0.01   Kurtosis:                        14.10
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    


```python
# ARIMA Forecast
arima_forecast = arima_fit.forecast(
    steps = len(test)
)
```


```python
# Evaluate ARIMA 
arima_results = evaluate_forecast(
    test[target],
    arima_forecast,
    'ARIMA Static'
)

arima_results
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
      <th>Model</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>Directional Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ARIMA Static</td>
      <td>0.007572</td>
      <td>0.010641</td>
      <td>0.537802</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Complete the comparison table:
results = pd.concat([
    benchmark_results, 
    arima_results,
], ignore_index = True
    )

results.sort_values('RMSE')
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
      <th>Model</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>Directional Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>ARIMA Static</td>
      <td>0.007572</td>
      <td>0.010641</td>
      <td>0.537802</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Historical Mean</td>
      <td>0.007573</td>
      <td>0.010642</td>
      <td>0.537023</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Zero Return</td>
      <td>0.007586</td>
      <td>0.010649</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Naive Momentum</td>
      <td>0.010987</td>
      <td>0.015177</td>
      <td>0.500390</td>
    </tr>
  </tbody>
</table>
</div>




```python
# PLot ARIMA results versus benchmarks
comparison['ARIMA Static'] = arima_forecast.values

comparison.iloc[:250].plot(
    title='Forecast Comparison'
)
```




```python
arima_rmse = arima_results['RMSE'].iloc[0]
arima_mae = arima_results['MAE'].iloc[0]
arima_dir_acc = arima_results['Directional Accuracy'].iloc[0]

# Margin between ARIMA and the historical mean, computed once and reused
# wherever the comparison is described.
#
# The absolute gap is uninformative on its own: RMSE here is of order 1e-2, so
# a difference of 1e-7 is a different kind of statement from a difference of
# 1e-3. The relative gap is the quantity that supports a claim about whether
# the two models are distinguishable.
#
# The 1e-4 threshold is a judgement, not a test. It says a relative difference
# below one hundredth of a percent is smaller than the precision this
# evaluation can support, given a single test window and a noisy target. It is
# stated explicitly rather than left implicit in a word like "identical".

rmse_margin = hist_mean_rmse - arima_rmse          # positive: ARIMA lower
rmse_margin_rel = abs(rmse_margin) / hist_mean_rmse
rmse_effective_tie = rmse_margin_rel < 1e-4
rmse_leader = 'ARIMA(1,0,1)' if rmse_margin > 0 else 'the historical mean'
rmse_parts = 1 / rmse_margin_rel if rmse_margin_rel > 0 else float('inf')

print(f'ARIMA RMSE          : {arima_rmse:.9f}')
print(f'Historical mean RMSE: {hist_mean_rmse:.9f}')
print(f'Absolute margin     : {rmse_margin:+.2e} (lower is better)')
print(f'Relative margin     : {rmse_margin_rel:.2e} '
      f'(one part in {rmse_parts:,.0f})')
print(f'Nominal leader      : {rmse_leader}')
print(f'Effective tie       : {rmse_effective_tie}')
```

    ARIMA RMSE          : 0.010641499
    Historical mean RMSE: 0.010641615
    Absolute margin     : +1.17e-07 (lower is better)
    Relative margin     : 1.10e-05 (one part in 91,286)
    Nominal leader      : ARIMA(1,0,1)
    Effective tie       : True
    


```python
display(Markdown(f"""
## Initial ARIMA results

ARIMA(1,0,1) produced an RMSE of {arima_rmse:.6f}, against
{hist_mean_rmse:.6f} for the historical mean forecast.

| Model | RMSE |
|---|---|
| ARIMA Static | {arima_rmse:.6f} |
| Historical Mean | {hist_mean_rmse:.6f} |
| Zero Return | {zero_rmse:.6f} |
| Naive Momentum | {naive_rmse:.6f} |

{rmse_leader} holds the lower figure, by {rmse_margin_rel:.2e} in relative
terms — one part in {rmse_parts:,.0f} of the value itself.
{'A margin that small is not a performance difference. It is the two models producing the same forecast to within the precision this evaluation supports, which is what a constant-mean ARIMA does when the autoregressive structure it fits is not significant.' if rmse_effective_tie else 'That margin is large enough to report as an ordering, though a single test window is thin evidence for it.'}

This matches the diagnostics from Notebook 02. Daily log returns are
stationary, but the ACF and PACF showed very weak autocorrelation, so the
model converges toward the long-run average return rather than extracting a
predictive signal.

Classical autoregressive models show limited ability to forecast daily market
direction here. Any more complex model needs to clear the historical mean
baseline to justify the added complexity.
"""))
```



## Initial ARIMA results

ARIMA(1,0,1) produced an RMSE of 0.010641, against
0.010642 for the historical mean forecast.

| Model | RMSE |
|---|---|
| ARIMA Static | 0.010641 |
| Historical Mean | 0.010642 |
| Zero Return | 0.010649 |
| Naive Momentum | 0.015177 |

ARIMA(1,0,1) holds the lower figure, by 1.10e-05 in relative
terms — one part in 91,286 of the value itself.
A margin that small is not a performance difference. It is the two models producing the same forecast to within the precision this evaluation supports, which is what a constant-mean ARIMA does when the autoregressive structure it fits is not significant.

This matches the diagnostics from Notebook 02. Daily log returns are
stationary, but the ACF and PACF showed very weak autocorrelation, so the
model converges toward the long-run average return rather than extracting a
predictive signal.

Classical autoregressive models show limited ability to forecast daily market
direction here. Any more complex model needs to clear the historical mean
baseline to justify the added complexity.



# ARIMA Walk-Forward Forecast

### Walk-Forward Validation (Rolling Forecast)

Instead of fitting the model once and forecasting the entire test set, we use a **walk-forward** approach:

- At each time step, the model is retrained using all available data up to that point.
- We forecast only the next day.
- The actual value is then added to the training history.

This method simulates real-world forecasting conditions more realistically than a single static fit, as it allows the model to adapt to new information over time.


```python
history = train[target].copy()

predictions = []

for t in test.index:

    actual_value = test.loc[t, target]

    model = ARIMA(history, order=(1, 0, 1))
    fitted = model.fit()

    forecast = fitted.forecast(steps=1)

    predictions.append(forecast.iloc[0])

    history = pd.concat([
        history,
        pd.Series([actual_value], index=[t])
    ])
```


```python
# Covert it to series
arima_walkforward = pd.Series(
    predictions,
    index = test.index
 )
```


```python
# Evaluate ARIMA Walk-Forward
arima_walk_results = evaluate_forecast(
    test[target],
    arima_walkforward,
    'ARIMA Walk-Forward'
)
```


```python
walkfwd_rmse = arima_walk_results['RMSE'].iloc[0]
walkfwd_mae = arima_walk_results['MAE'].iloc[0]
walkfwd_dir_acc = arima_walk_results['Directional Accuracy'].iloc[0]
```


```python
# Complete the comparison table:
results = pd.concat([
    results, 
    arima_walk_results,
], ignore_index = True
    )

results.sort_values('RMSE')
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
      <th>Model</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>Directional Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>ARIMA Static</td>
      <td>0.007572</td>
      <td>0.010641</td>
      <td>0.537802</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Historical Mean</td>
      <td>0.007573</td>
      <td>0.010642</td>
      <td>0.537023</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Zero Return</td>
      <td>0.007586</td>
      <td>0.010649</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ARIMA Walk-Forward</td>
      <td>0.007663</td>
      <td>0.010690</td>
      <td>0.507405</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Naive Momentum</td>
      <td>0.010987</td>
      <td>0.015177</td>
      <td>0.500390</td>
    </tr>
  </tbody>
</table>
</div>




```python
display(Markdown(f"""
## ARIMA evaluation

ARIMA(1,0,1) was evaluated with both a static forecast (RMSE = {arima_rmse:.6f},
directional accuracy = {arima_dir_acc:.2%}) and an expanding-window
walk-forward setup (RMSE = {walkfwd_rmse:.6f}, directional accuracy =
{walkfwd_dir_acc:.2%}). Neither approach beat the historical mean
(RMSE = {hist_mean_rmse:.6f}).

This is consistent with the diagnostics: weak autocorrelation in daily log
returns, non-significant AR and MA coefficients, and limited evidence of
short-term return predictability. Daily S&P 500 log returns appear to contain
little linear predictive structure — the stronger patterns sit in volatility,
not direction.
"""))
```



## ARIMA evaluation

ARIMA(1,0,1) was evaluated with both a static forecast (RMSE = 0.010641,
directional accuracy = 53.78%) and an expanding-window
walk-forward setup (RMSE = 0.010690, directional accuracy =
50.74%). Neither approach beat the historical mean
(RMSE = 0.010642).

This is consistent with the diagnostics: weak autocorrelation in daily log
returns, non-significant AR and MA coefficients, and limited evidence of
short-term return predictability. Daily S&P 500 log returns appear to contain
little linear predictive structure — the stronger patterns sit in volatility,
not direction.



# ARIMA residual diagnostics


```python
residuals = arima_fit.resid
```


```python
# Residual Distribution

residuals.plot.hist(
    bins = 50,
    title = 'ARIMA Residual Distribution'
)
```




```python
# Residuals ACF 
plot_acf(residuals, lags = 40);
```


    
![png](04_Baseline_Forecasting_Models_files/04_Baseline_Forecasting_Models_42_0.png)
    



```python
# Ljung-Bos test on Residuals 
acorr_ljungbox(
    residuals,
    lags = [10, 20],
    return_df = True
)
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
      <th>lb_stat</th>
      <th>lb_pvalue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>29.076686</td>
      <td>1.210916e-03</td>
    </tr>
    <tr>
      <th>20</th>
      <td>90.675256</td>
      <td>5.647005e-11</td>
    </tr>
  </tbody>
</table>
</div>




```python
jarque_bera(residuals)
```




    SignificanceResult(statistic=np.float64(26599.82957069832), pvalue=np.float64(0.0))




```python
lb_results = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
lb_q10 = lb_results.loc[10, 'lb_stat']
lb_p10 = lb_results.loc[10, 'lb_pvalue']
lb_q20 = lb_results.loc[20, 'lb_stat']
lb_p20 = lb_results.loc[20, 'lb_pvalue']

jb_stat, jb_pvalue = jarque_bera(residuals)
```


```python
display(Markdown(f"""
## ARIMA residual diagnostics

Most residual autocorrelations fall within the confidence bounds, though
several isolated lags remain significant.

The Ljung-Box test rejects independence at lag 10 (Q = {lb_q10:.2f},
p {'< 0.001' if lb_p10 < 0.001 else f'= {lb_p10:.4f}'}) and lag 20
(Q = {lb_q20:.2f}, p {'< 0.001' if lb_p20 < 0.001 else f'= {lb_p20:.4f}'}).
The statistic strengthening at longer lags points to volatility clustering
rather than a short-term autoregressive artefact.

The Jarque-Bera test rejects normality (statistic = {jb_stat:,.2f},
p {'≈ 0' if jb_pvalue < 0.0001 else f'= {jb_pvalue:.4f}'}), confirming fat
tails persist after fitting ARIMA(1,0,1). The model removes part of the linear
dependence in returns but doesn't fully whiten the residuals — what's left is
concentrated in volatility, not direction.
"""))
```



## ARIMA residual diagnostics

Most residual autocorrelations fall within the confidence bounds, though
several isolated lags remain significant.

The Ljung-Box test rejects independence at lag 10 (Q = 29.08,
p = 0.0012) and lag 20
(Q = 90.68, p < 0.001).
The statistic strengthening at longer lags points to volatility clustering
rather than a short-term autoregressive artefact.

The Jarque-Bera test rejects normality (statistic = 26,599.83,
p ≈ 0), confirming fat
tails persist after fitting ARIMA(1,0,1). The model removes part of the linear
dependence in returns but doesn't fully whiten the residuals — what's left is
concentrated in volatility, not direction.




```python
# Plotting the autocorrelation volatility of the residuals
plot_acf(residuals**2, lags=40);

```


    
![png](04_Baseline_Forecasting_Models_files/04_Baseline_Forecasting_Models_47_0.png)
    


## Squared residual diagnostics

The ACF of the squared ARIMA residuals shows statistically significant 
autocorrelation across multiple lags, with correlations gradually decaying 
over time. This pattern is characteristic of volatility clustering and confirms 
that variance persistence remains after modelling the conditional mean.

This result is consistent with Notebook 02, where squared log returns exhibited 
substantially stronger autocorrelation than the returns themselves. The structure 
that ARIMA leaves unexplained is in the variance, not the mean — which motivates 
GARCH-family modelling in Notebook 05.

## Why the Engineered Features Were Not Used

Notebook 03 produced a feature set containing lagged returns, rolling statistics, technical indicators, calendar effects, and volatility regime signals.

These features were intentionally excluded from the models in this notebook.

The objective of Notebook 04 is to establish a pure univariate forecasting benchmark using only the historical behaviour of daily S&P 500 log returns. This provides a clean reference point for evaluating whether more sophisticated approaches can add predictive value.

In later notebooks, these engineered features will be introduced as exogenous inputs through machine learning and regression-based forecasting models. Their performance will be evaluated against the classical benchmarks established here.

# SARIMA Model


```python
# Using Seasonal Period = 5 (trading week)
sarima_model = SARIMAX(
    train[target],
    order = (1, 0, 1),   # Non-Seasonal 
    seasonal_order = (1, 0, 1, 5),   # Seasonal (Weekly)
    enforce_stationarity = False,
    enforce_invertibility = False
)

sarima_fit = sarima_model.fit(disp=False)

print(sarima_fit.summary())
```

                                         SARIMAX Results                                     
    =========================================================================================
    Dep. Variable:                       log_returns   No. Observations:                 5129
    Model:             SARIMAX(1, 0, 1)x(1, 0, 1, 5)   Log Likelihood               15262.243
    Date:                           Mon, 17 Aug 2026   AIC                         -30514.486
    Time:                                   23:41:33   BIC                         -30481.780
    Sample:                                        0   HQIC                        -30503.037
                                              - 5129                                         
    Covariance Type:                             opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1         -0.0548      0.046     -1.196      0.232      -0.145       0.035
    ma.L1         -0.0648      0.047     -1.390      0.165      -0.156       0.027
    ar.S.L5        0.1930      0.239      0.808      0.419      -0.275       0.661
    ma.S.L5       -0.2126      0.238     -0.895      0.371      -0.678       0.253
    sigma2         0.0002   1.21e-06    125.329      0.000       0.000       0.000
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.04   Jarque-Bera (JB):             27113.80
    Prob(Q):                              0.85   Prob(JB):                         0.00
    Heteroskedasticity (H):               1.14   Skew:                            -0.61
    Prob(H) (two-sided):                  0.01   Kurtosis:                        14.20
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    


```python
# Forecast 
sarima_forecast = sarima_fit.forecast(
    steps = len(test)
)

# Evaluate Results 
sarima_results = evaluate_forecast(
    test[target],
    sarima_forecast,
    'SARIMA(1,0,1)(1,0,1,5)'
)
```


```python
# Add Results to results data frame
results = pd.concat(
    [results, sarima_results],
    ignore_index = True
)
results
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
      <th>Model</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>Directional Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Historical Mean</td>
      <td>0.007573</td>
      <td>0.010642</td>
      <td>0.537023</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Naive Momentum</td>
      <td>0.010987</td>
      <td>0.015177</td>
      <td>0.500390</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Zero Return</td>
      <td>0.007586</td>
      <td>0.010649</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ARIMA Static</td>
      <td>0.007572</td>
      <td>0.010641</td>
      <td>0.537802</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ARIMA Walk-Forward</td>
      <td>0.007663</td>
      <td>0.010690</td>
      <td>0.507405</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SARIMA(1,0,1)(1,0,1,5)</td>
      <td>0.007586</td>
      <td>0.010649</td>
      <td>0.463757</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Seasonal parameter labels vary with the seasonal period and statsmodels
# version, so locate them by their '.S.' infix rather than hardcoding keys
_seasonal_ar = [p for p in sarima_fit.pvalues.index if p.startswith('ar.S.')]
_seasonal_ma = [p for p in sarima_fit.pvalues.index if p.startswith('ma.S.')]
sarima_seasonal_ar_pvalue = sarima_fit.pvalues[_seasonal_ar[0]]
sarima_seasonal_ma_pvalue = sarima_fit.pvalues[_seasonal_ma[0]]

arima_aic = arima_fit.aic
sarima_aic = sarima_fit.aic

sarima_rmse = sarima_results['RMSE'].iloc[0]
sarima_mae = sarima_results['MAE'].iloc[0]
sarima_dir_acc = sarima_results['Directional Accuracy'].iloc[0]
```


```python
# Is SARIMA's directional accuracy distinguishable from a coin flip?
#
# Under a null of no directional skill the sign is correct with probability
# 0.5, so the standard error of the observed proportion is sqrt(0.25 / n).
# Reporting the z-score rather than the raw percentage keeps the claim
# proportionate to the sample size behind it.

dir_acc_se = np.sqrt(0.25 / len(test))
sarima_dir_z = (sarima_dir_acc - 0.5) / dir_acc_se

print(f'SARIMA directional accuracy: {sarima_dir_acc:.2%}')
print(f'Standard error under H0    : {dir_acc_se:.4f}')
print(f'z vs 50%                   : {sarima_dir_z:+.2f}')

display(Markdown(f"""
## SARIMA evaluation

SARIMA(1,0,1)(1,0,1,5) tested whether weekly seasonality (5 trading days)
improves forecasting performance. The seasonal AR(5) coefficient came in at
p = {sarima_seasonal_ar_pvalue:.3f} and the seasonal MA(5) coefficient at
p = {sarima_seasonal_ma_pvalue:.3f}, both non-significant. AIC rose from
{arima_aic:,.2f} (ARIMA) to {sarima_aic:,.2f} (SARIMA) — a worse fit despite
the added parameters.

Notebook 02 found some calendar effects, including stronger average returns
on Tuesdays and Wednesdays and weaker September performance. The seasonal
terms here don't pick that up, which suggests any weekly seasonality is weak
relative to overall noise in daily log returns.

### On the directional accuracy

SARIMA called direction correctly on {sarima_dir_acc:.2%} of test days,
{'below' if sarima_dir_acc < 0.5 else 'above'} the 50% a coin flip would give.
Across {len(test):,} test observations the standard error of that proportion
under a no-skill null is {dir_acc_se:.4f}, placing the observed figure
{abs(sarima_dir_z):.1f} standard errors {'below' if sarima_dir_z < 0 else 'above'}
50% (z = {sarima_dir_z:+.2f}).

{'That is nominally significant, and it is worth being explicit about what it is not. Directional accuracy is a secondary statistic here: RMSE and AIC are the selection criteria, and on both this specification lost. It is one figure among several inspected across seven models, with no correction for multiple comparisons applied, and it comes from a model whose seasonal coefficients were themselves non-significant. A sub-coin-flip hit rate produced by a specification that fits no real structure is the sign of a forecast whose direction is arbitrary, not a tradable inverse signal. Treating it as one would mean betting against a coefficient the data says is indistinguishable from zero.' if abs(sarima_dir_z) > 1.96 else 'That is within sampling error of a coin flip, consistent with a specification that fits no real directional structure.'}

RMSE reinforces the reading: at {sarima_rmse:.6f} against
{hist_mean_rmse:.6f} for the historical mean, the model shows no magnitude
edge to accompany the sign pattern. RMSE is driven by the size of errors
rather than their direction, so the two statistics can disagree without either
being wrong.

ARIMA(1,0,1) remains the preferred specification on both significance and
AIC. The calendar effects from exploratory analysis are likely better
captured through explicit calendar features (Notebook 03) than a seasonal
ARIMA term.
"""))
```

    SARIMA directional accuracy: 46.38%
    Standard error under H0    : 0.0140
    z vs 50%                   : -2.60
    



## SARIMA evaluation

SARIMA(1,0,1)(1,0,1,5) tested whether weekly seasonality (5 trading days)
improves forecasting performance. The seasonal AR(5) coefficient came in at
p = 0.419 and the seasonal MA(5) coefficient at
p = 0.371, both non-significant. AIC rose from
-30,556.10 (ARIMA) to -30,514.49 (SARIMA) — a worse fit despite
the added parameters.

Notebook 02 found some calendar effects, including stronger average returns
on Tuesdays and Wednesdays and weaker September performance. The seasonal
terms here don't pick that up, which suggests any weekly seasonality is weak
relative to overall noise in daily log returns.

### On the directional accuracy

SARIMA called direction correctly on 46.38% of test days,
below the 50% a coin flip would give.
Across 1,283 test observations the standard error of that proportion
under a no-skill null is 0.0140, placing the observed figure
2.6 standard errors below
50% (z = -2.60).

That is nominally significant, and it is worth being explicit about what it is not. Directional accuracy is a secondary statistic here: RMSE and AIC are the selection criteria, and on both this specification lost. It is one figure among several inspected across seven models, with no correction for multiple comparisons applied, and it comes from a model whose seasonal coefficients were themselves non-significant. A sub-coin-flip hit rate produced by a specification that fits no real structure is the sign of a forecast whose direction is arbitrary, not a tradable inverse signal. Treating it as one would mean betting against a coefficient the data says is indistinguishable from zero.

RMSE reinforces the reading: at 0.010649 against
0.010642 for the historical mean, the model shows no magnitude
edge to accompany the sign pattern. RMSE is driven by the size of errors
rather than their direction, so the two statistics can disagree without either
being wrong.

ARIMA(1,0,1) remains the preferred specification on both significance and
AIC. The calendar effects from exploratory analysis are likely better
captured through explicit calendar features (Notebook 03) than a seasonal
ARIMA term.



# Prophet Model


```python
# Prophet requires columns ds and y 
prophet_train = train[[target]].reset_index()

prophet_train.columns = ['ds', 'y']

prophet_train.head()
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
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001-02-13</td>
      <td>-0.008690</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001-02-14</td>
      <td>-0.002186</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001-02-15</td>
      <td>0.008091</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001-02-16</td>
      <td>-0.019086</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001-02-20</td>
      <td>-0.017509</td>
    </tr>
  </tbody>
</table>
</div>




```python
prophet_model = Prophet(
    daily_seasonality = False,
    weekly_seasonality = True,
    yearly_seasonality= True
    )

prophet_model.fit(prophet_train)
```

    23:41:34 - cmdstanpy - INFO - Chain [1] start processing
    23:41:34 - cmdstanpy - INFO - Chain [1] done processing
    




    <prophet.forecaster.Prophet at 0x1f752669250>




```python
# Create Future dates
# We use trading days as market doesn't trade on weekends
future = prophet_model.make_future_dataframe(
    periods = len(test),
    freq = 'B'
)

```


```python
# Forecast
prophet_forecast_full = prophet_model.predict(future)

prophet_forecast = (
    prophet_forecast_full['yhat']
    .iloc[-len(test):]
    .values
)

```


```python
# Evaluate Results 
prophet_results = evaluate_forecast(
    test[target],
    prophet_forecast,
    'Prophet'
)

prophet_results
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
      <th>Model</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>Directional Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Prophet</td>
      <td>0.007612</td>
      <td>0.010694</td>
      <td>0.522993</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add Prophet Results to the comparison table 
results = pd.concat(
    [results, prophet_results],
    ignore_index = True
)

results.sort_values('RMSE')
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
      <th>Model</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>Directional Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>ARIMA Static</td>
      <td>0.007572</td>
      <td>0.010641</td>
      <td>0.537802</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Historical Mean</td>
      <td>0.007573</td>
      <td>0.010642</td>
      <td>0.537023</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SARIMA(1,0,1)(1,0,1,5)</td>
      <td>0.007586</td>
      <td>0.010649</td>
      <td>0.463757</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Zero Return</td>
      <td>0.007586</td>
      <td>0.010649</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ARIMA Walk-Forward</td>
      <td>0.007663</td>
      <td>0.010690</td>
      <td>0.507405</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Prophet</td>
      <td>0.007612</td>
      <td>0.010694</td>
      <td>0.522993</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Naive Momentum</td>
      <td>0.010987</td>
      <td>0.015177</td>
      <td>0.500390</td>
    </tr>
  </tbody>
</table>
</div>




```python
prophet_rmse = prophet_results['RMSE'].iloc[0]
prophet_mae = prophet_results['MAE'].iloc[0]
prophet_dir_acc = prophet_results['Directional Accuracy'].iloc[0]
```


```python
display(Markdown(f"""
## Prophet evaluation

Prophet was included as a benchmark given its common use for trend and
seasonality forecasting. RMSE came in at {prophet_rmse:.6f}, MAE at
{prophet_mae:.6f}, and directional accuracy at {prophet_dir_acc:.2%} — none
of which improved on the historical mean (RMSE = {hist_mean_rmse:.6f}).

This matches the return characteristics from Notebook 02. Unlike business
metrics such as revenue or web traffic, daily returns carry little persistent
trend and weak seasonality. Prophet is built to model smooth trends and
recurring seasonal patterns; daily financial returns are dominated by noise
and short-lived shocks, leaving little of that structure for it to find.
"""))
```



## Prophet evaluation

Prophet was included as a benchmark given its common use for trend and
seasonality forecasting. RMSE came in at 0.010694, MAE at
0.007612, and directional accuracy at 52.30% — none
of which improved on the historical mean (RMSE = 0.010642).

This matches the return characteristics from Notebook 02. Unlike business
metrics such as revenue or web traffic, daily returns carry little persistent
trend and weak seasonality. Prophet is built to model smooth trends and
recurring seasonal patterns; daily financial returns are dominated by noise
and short-lived shocks, leaving little of that structure for it to find.




```python
results_sorted = results.sort_values('RMSE').reset_index(drop=True)
best_model = results_sorted.iloc[0]['Model']
best_rmse = results_sorted.iloc[0]['RMSE']
runner_up = results_sorted.iloc[1]['Model']
runner_up_rmse = results_sorted.iloc[1]['RMSE']
top_two_rel = abs(best_rmse - runner_up_rmse) / runner_up_rmse

display(Markdown(f"""
# Notebook 04 conclusions

This notebook evaluated classical time series models as forecasting
benchmarks for daily S&P 500 log returns.

{results_sorted.to_markdown(index=False)}

No classical model meaningfully outperformed the historical mean benchmark
(RMSE = {hist_mean_rmse:.6f}). {best_model} sits at the top of the table with
RMSE {best_rmse:.6f}, ahead of {runner_up} by {top_two_rel:.2e} in relative
terms.
{'Reading that ordering as a result would be a mistake: the gap is far below the precision a single 252-day-scale test window on a noisy target can resolve, so the table records which model happened to land marginally lower rather than which forecasts better.' if top_two_rel < 1e-4 else 'That gap is large enough to state as an ordering, though one test window remains thin evidence.'}
The finding of this notebook is the absence of a margin, not the identity of
the nominal leader.

ARIMA(1,0,1) captured only limited linear structure. SARIMA's weekly seasonal
coefficients were non-significant (p = {sarima_seasonal_ar_pvalue:.3f} and
p = {sarima_seasonal_ma_pvalue:.3f}) and the seasonal term didn't improve fit
(AIC {sarima_aic:,.2f} vs {arima_aic:,.2f}); its directional accuracy of
{sarima_dir_acc:.2%} is discussed in that section. Prophet found no useful
trend or seasonal signal beyond what the simpler benchmarks already captured.
The historical mean held up against all of them.

Residual diagnostics reinforced the same picture: Ljung-Box rejected
independence at lag 10 (Q = {lb_q10:.2f}) and lag 20 (Q = {lb_q20:.2f}),
strengthening at longer lags. Jarque-Bera rejected normality
(statistic = {jb_stat:,.2f}), confirming fat tails survive the fit. The ACF
of squared residuals decayed gradually across 40 lags — volatility
clustering, not noise.

Daily S&P 500 log returns carry very little predictable structure in their
conditional mean, and increasing model complexity didn't change that. Return
direction is difficult to predict, but market risk is not randomly distributed
through time — the remaining structure sits in conditional variance. That
shifts the focus from forecasting direction to forecasting risk, and motivates
GARCH-family modelling in Notebook 05.
"""))
```



# Notebook 04 conclusions

This notebook evaluated classical time series models as forecasting
benchmarks for daily S&P 500 log returns.

| Model                  |        MAE |      RMSE |   Directional Accuracy |
|:-----------------------|-----------:|----------:|-----------------------:|
| ARIMA Static           | 0.00757195 | 0.0106415 |               0.537802 |
| Historical Mean        | 0.00757267 | 0.0106416 |               0.537023 |
| SARIMA(1,0,1)(1,0,1,5) | 0.00758564 | 0.0106489 |               0.463757 |
| Zero Return            | 0.00758631 | 0.0106489 |               0        |
| ARIMA Walk-Forward     | 0.00766289 | 0.0106905 |               0.507405 |
| Prophet                | 0.00761161 | 0.0106939 |               0.522993 |
| Naive Momentum         | 0.0109871  | 0.0151765 |               0.50039  |

No classical model meaningfully outperformed the historical mean benchmark
(RMSE = 0.010642). ARIMA Static sits at the top of the table with
RMSE 0.010641, ahead of Historical Mean by 1.10e-05 in relative
terms.
Reading that ordering as a result would be a mistake: the gap is far below the precision a single 252-day-scale test window on a noisy target can resolve, so the table records which model happened to land marginally lower rather than which forecasts better.
The finding of this notebook is the absence of a margin, not the identity of
the nominal leader.

ARIMA(1,0,1) captured only limited linear structure. SARIMA's weekly seasonal
coefficients were non-significant (p = 0.419 and
p = 0.371) and the seasonal term didn't improve fit
(AIC -30,514.49 vs -30,556.10); its directional accuracy of
46.38% is discussed in that section. Prophet found no useful
trend or seasonal signal beyond what the simpler benchmarks already captured.
The historical mean held up against all of them.

Residual diagnostics reinforced the same picture: Ljung-Box rejected
independence at lag 10 (Q = 29.08) and lag 20 (Q = 90.68),
strengthening at longer lags. Jarque-Bera rejected normality
(statistic = 26,599.83), confirming fat tails survive the fit. The ACF
of squared residuals decayed gradually across 40 lags — volatility
clustering, not noise.

Daily S&P 500 log returns carry very little predictable structure in their
conditional mean, and increasing model complexity didn't change that. Return
direction is difficult to predict, but market risk is not randomly distributed
through time — the remaining structure sits in conditional variance. That
shifts the focus from forecasting direction to forecasting risk, and motivates
GARCH-family modelling in Notebook 05.




```python
# Save metrics from this notebook for furture call
metrics_path = Path('../data/locked_metrics.json')
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
 
metrics['notebook_04'] = {
    'arima_order': '1,0,1',
    'arima_rmse': float(arima_rmse),
    'historical_mean_rmse': float(hist_mean_rmse),
}
 
metrics_path.write_text(json.dumps(metrics, indent=2))
print(f"Exported notebook_04 metrics to {metrics_path.resolve()}")
for k, v in metrics['notebook_04'].items():
    print(f"  {k}: {v}")
 
```

    Exported notebook_04 metrics to C:\Users\Mena\Documents\Python\sp500-market-intelligence\data\locked_metrics.json
      arima_order: 1,0,1
      arima_rmse: 0.010641498724367367
      historical_mean_rmse: 0.01064161529907483
    
