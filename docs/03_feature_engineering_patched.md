# Feature Engineering & Technical Indicators for S&P 500

This notebook transforms the cleaned S&P 500 dataset into a feature matrix for
volatility modelling, regime detection, and forecasting.

The emphasis is on economic meaning rather than indicator count. Features are
selected in response to the statistical properties identified during diagnostics:
fat tails, volatility clustering, and weak directional dependence in returns.

## Feature Categories

1. **Lag Features** — Short-term temporal dependencies
2. **Rolling Statistics** — Changing market conditions
3. **Technical Indicators** — Momentum and volatility measures
4. **Calendar Features** — Seasonal and temporal effects
5. **Regime Features** — Volatility states and market regimes

These features will support classical, GARCH, and machine learning models in later notebooks.

## Load Cleaned Dataset


```python
# Import the needed libraries
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import Markdown, display
import json
from pathlib import Path
import os

pd.options.plotting.backend = 'plotly'
pio.templates.default = 'plotly_dark'
os.makedirs('../reports/figures', exist_ok=True)
```


```python
# Load cleaned dataset from EDA
df = pd.read_parquet('../data/sp500_cleaned.parquet').copy()

# Preview dataset
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
      <th>Price</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>log_returns</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>2000-01-04</th>
      <td>1455.219971</td>
      <td>1455.219971</td>
      <td>1397.430054</td>
      <td>1399.420044</td>
      <td>1009000000</td>
      <td>-0.039099</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>1399.420044</td>
      <td>1413.270020</td>
      <td>1377.680054</td>
      <td>1402.109985</td>
      <td>1085500000</td>
      <td>0.001920</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>1402.109985</td>
      <td>1411.900024</td>
      <td>1392.099976</td>
      <td>1403.449951</td>
      <td>1092300000</td>
      <td>0.000955</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>1403.449951</td>
      <td>1441.469971</td>
      <td>1400.729980</td>
      <td>1441.469971</td>
      <td>1225200000</td>
      <td>0.026730</td>
    </tr>
    <tr>
      <th>2000-01-10</th>
      <td>1441.469971</td>
      <td>1464.359985</td>
      <td>1441.469971</td>
      <td>1457.599976</td>
      <td>1064800000</td>
      <td>0.011128</td>
    </tr>
  </tbody>
</table>
</div>




```python

# Dataset validation summary function
def dataset_summary(df):
    return {
        'rows': len(df),
        'start': df.index.min().date(),
        'end': df.index.max().date()
    }

summary = dataset_summary(df)

print(f"Rows: {summary['rows']:,}")
print(f"Date range: {summary['start']} → {summary['end']}")
```

    Rows: 6,692
    Date range: 2000-01-04 → 2026-08-14
    

## Loaded cleaned dataset

The dataset loaded here is the validated output from the EDA notebook. Data
integrity checks, the zero-volume anomaly removal, T-1 enforcement, and log
return calculations are already complete. Feature engineering starts from a
consistent baseline.

The file is stored in Parquet format to preserve column types and schema across
sessions.


```python
display(Markdown(f"""
## Dataset validation

- **Rows:** {summary['rows']:,}
- **Temporal coverage:** {summary['start']} → {summary['end']}

Confirms the loaded dataset matches the expected EDA output before feature
construction begins.
"""))
```



## Dataset validation

- **Rows:** 6,692
- **Temporal coverage:** 2000-01-04 → 2026-08-14

Confirms the loaded dataset matches the expected EDA output before feature
construction begins.




```python
# Load the diagnostics figures this notebook cites.
#
# Two markdown sections below quote the ARCH-LM statistic as the justification
# for the lag structure and the regime features. Those sections previously
# carried the number typed in by hand, and it fell out of date the first time
# NB02 was re-run on a longer sample. Reading by key means the prose cannot
# state a figure the diagnostics notebook did not produce.
#
# The lag count travels with the statistic. NB02 runs the test at 10 lags and
# NB05 at 20, which produce different values from the same series, so a bare
# LM figure is ambiguous about which test it came from.

metrics_path = Path('../data/locked_metrics.json')

if not metrics_path.exists():
    raise FileNotFoundError(
        f'{metrics_path} not found. Run Notebook 02 before this notebook: it '
        'writes the notebook_02 block this notebook reads.'
    )

metrics = json.loads(metrics_path.read_text())

if 'notebook_02' not in metrics:
    raise KeyError(
        "locked_metrics.json has no 'notebook_02' block. Run Notebook 02 to "
        'completion, including its final export cell.'
    )

nb02 = metrics['notebook_02']
arch_lm = nb02['arch_lm_stat']
arch_lm_p = nb02['arch_lm_p']
arch_lm_nlags = nb02['arch_lm_nlags']

print(f'Loaded from {metrics_path.name}:')
print(f'  ARCH-LM statistic : {arch_lm:,.2f} ({arch_lm_nlags} lags)')
print(f'  ARCH-LM p-value   : {arch_lm_p:.3g}')
```

    Loaded from locked_metrics.json:
      ARCH-LM statistic : 1,786.70 (10 lags)
      ARCH-LM p-value   : 0
    

# 1-Lag Features


```python
lags = [1, 2, 3, 5]

for lag in lags:
    df[f'return_lag_{lag}'] = df['log_returns'].shift(lag)
    
# Stationary alternatives to raw price lags
df['close_pct_lag_1'] = df['Close'].pct_change().shift(1)
df['close_pct_lag_2'] = df['Close'].pct_change().shift(2)

# Additional Lagged features
df['range_lag_1'] = (df['High'] - df['Low']).shift(1)
df['volume_lag_1'] = df['Volume'].shift(1)

lag_cols = [c for c in df.columns if '_lag_' in c]
print(f"Created {len(lag_cols)} lag features")
```

    Created 8 lag features
    


```python
display(Markdown(f"""
## Lag features short-term temporal memory

Lag features give models access to recent market information through delayed
versions of returns, price movements, and trading activity.

Lag selection was guided by PACF analysis, short-term dependence structure, and
redundancy. PACF on raw returns showed limited directional persistence,
concentrated in the earliest lags. Squared returns told a different story:
ACF and PACF diagnostics revealed significant short-term volatility dependence
— consistent with the ARCH-LM result
(LM = {arch_lm:,.2f}, {arch_lm_nlags} lags,
p {'< 0.001' if arch_lm_p < 0.001 else f'= {arch_lm_p:.4f}'})
from the diagnostics notebook.

**Features created:**

- `return_lag_1`, `return_lag_2`, `return_lag_3`, `return_lag_5` — recent
  return history
- `close_pct_lag_1`, `close_pct_lag_2` — lagged percentage price changes,
  preferred over raw price lags to avoid non-stationarity
- `range_lag_1` — previous day intraday range (High − Low), a proxy for
  recent volatility intensity
- `volume_lag_1` — previous day volume, providing liquidity context

Return direction shows weak predictability. Volatility and risk exhibit
measurable memory. The lag structure reflects both observations: short-term
information across direction, price movement, volatility, and participation
— without unnecessary lag depth.
"""))
```



## Lag features short-term temporal memory

Lag features give models access to recent market information through delayed
versions of returns, price movements, and trading activity.

Lag selection was guided by PACF analysis, short-term dependence structure, and
redundancy. PACF on raw returns showed limited directional persistence,
concentrated in the earliest lags. Squared returns told a different story:
ACF and PACF diagnostics revealed significant short-term volatility dependence
— consistent with the ARCH-LM result
(LM = 1,786.70, 10 lags,
p < 0.001)
from the diagnostics notebook.

**Features created:**

- `return_lag_1`, `return_lag_2`, `return_lag_3`, `return_lag_5` — recent
  return history
- `close_pct_lag_1`, `close_pct_lag_2` — lagged percentage price changes,
  preferred over raw price lags to avoid non-stationarity
- `range_lag_1` — previous day intraday range (High − Low), a proxy for
  recent volatility intensity
- `volume_lag_1` — previous day volume, providing liquidity context

Return direction shows weak predictability. Volatility and risk exhibit
measurable memory. The lag structure reflects both observations: short-term
information across direction, price movement, volatility, and participation
— without unnecessary lag depth.



# 2-Rolling Statistics 


```python
windows = [5, 10, 21, 30, 60]

for w in windows:
    # Volatility measures
    df[f'vol_roll_{w}'] = df['log_returns'].rolling(window=w).std()

    # Mean dynamics
    df[f'return_mean_roll_{w}'] = df['log_returns'].rolling(window=w).mean()

    # Momentum (price relative to moving average)
    df[f'momentum_{w}'] = df['Close'] / df['Close'].rolling(window=w).mean() - 1


print(f'Created rolling features across {len(windows)} windows (5 to 60 days)')
```

    Created rolling features across 5 windows (5 to 60 days)
    

## Rolling statistics — multi-horizon market behaviour

Rolling features let models observe how market conditions evolve rather than
treating each observation as independent. Five windows were used: 5, 10, 21,
30, and 60 days.

These correspond to recognisable market horizons: one trading week, two weeks,
approximately one trading month, the standard rolling volatility window used in
risk reporting, and a medium-term regime window. Each window produces three
features: rolling volatility (`vol_roll_*`), rolling mean returns
(`return_mean_roll_*`), and a price-to-moving-average momentum measure
(`momentum_*`).

The momentum feature is calculated as:

$$\frac{Close_t}{MA_t} - 1$$

Positive values indicate price trading above its recent average; negative values
indicate relative weakness.

The statistical case for rolling features comes directly from Notebook 02.
Weak serial dependence in raw returns and strong volatility clustering mean the
most useful market information is in *how risk is changing*, not where prices
moved yesterday. Rolling features capture that evolving structure across multiple
horizons.

# 3-Technical Indicators


```python
# 1- Relative Strength Index (RSI)
def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi_14'] = rsi(df['Close'], 14)

# 2- MACD (Moving Average Convergence Divergence)
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['macd'] = ema12 - ema26
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']

# 3- Bollinger Bands 
df['bb_middle'] = df['Close'].rolling(window=20).mean()
df['bb_std'] = df['Close'].rolling(window=20).std()
df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

# 4- Average True Range (ATR)
high_low = df['High'] - df['Low']
high_close = np.abs(df['High'] - df['Close'].shift())
low_close = np.abs(df['Low'] - df['Close'].shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr_14'] = tr.rolling(window=14).mean()

print('Technical Indicators created: RSI, MACD, Bollinger Bands, ATR')

```

    Technical Indicators created: RSI, MACD, Bollinger Bands, ATR
    

## Technical indicators

Four indicators were added, each selected for what it contributes as an
explanatory feature rather than as a trading signal.

- **RSI (14)** — Quantifies momentum by comparing the magnitude of recent
  gains against losses. Useful for capturing overbought and oversold conditions.
- **MACD** — Tracks the spread between a 12-day and 26-day EMA, with a 9-day
  signal line. Represents trend momentum and directional shifts.
- **Bollinger Bands (20-day, 2σ)** — Express price relative to a rolling mean
  and standard deviation envelope. `bb_width` measures how far the bands are
  spread, making it a direct volatility proxy.
- **ATR (14)** — Averages the true range over 14 days, accounting for
  intraday movement and overnight gaps. Captures market range volatility
  independently of price direction.

RSI and MACD represent momentum and trend. Bollinger Bands and ATR capture
changing volatility conditions. Both are relevant given the volatility
clustering and conditional heteroskedasticity confirmed in diagnostics. Whether
these features improve forecasting accuracy is a question for the modelling
notebooks.

# 4- Calender and Seasonality Features


```python
# Basic Temporal Features
df['day_of_week'] = df.index.dayofweek # 0 = Monday, 6 = Sunday
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['is_month_end'] = df.index.is_month_end.astype(int)
df['is_quarter_end'] = df.index.is_quarter_end.astype(int)

# Day of the week dummies (one-hot-encoding)
dow_dummies = pd.get_dummies(
    df['day_of_week'],
    prefix='dow',
    drop_first=True,
    dtype=int
)

month_dummies = pd.get_dummies(
    df['month'],
    prefix='month',
    drop_first=True,
    dtype=int
)

df = pd.concat([df, dow_dummies, month_dummies], axis=1)

print("Calendar and seasonality features created")
print(f"Added {len(dow_dummies.columns) + len(month_dummies.columns)} dummy variables")
```

    Calendar and seasonality features created
    Added 15 dummy variables
    

## Calendar and seasonality features

Calendar features capture systematic temporal patterns — day-of-week effects
and month-of-year seasonality observed during exploratory analysis.

**Features created:**

- `day_of_week`, `month`, `quarter` — raw temporal indices
- `is_month_end`, `is_quarter_end` — binary flags for period boundaries
- `dow_*` — one-hot encoded day-of-week dummies (reference day dropped)
- `month_*` — one-hot encoded month dummies (reference month dropped)

Exploratory analysis showed Tuesday and Wednesday averaging slightly stronger
returns, and September exhibiting historically weaker performance. Institutional
activity — rebalancing cycles, reporting periods, month-end positioning, options
expiry — follows predictable timing. Calendar features give models a mechanism
to learn those patterns.

One-hot encoding is used rather than raw numeric calendar values. Treating
day-of-week or month as a continuous variable would impose an ordering that
doesn't exist economically. September is not numerically larger than February
in any meaningful sense.

Calendar effects are unlikely to be strong standalone signals, but combined with
volatility and momentum features, they may improve regime identification and
forecasting at the margin.

# 5- Regime Features


```python
# Volatility above 1-year historical average:
df['vol_above_avg'] = np.where(
    df['vol_roll_30'].rolling(252).mean().isna(),
    np.nan,
    (
        df['vol_roll_30']
        > df['vol_roll_30'].rolling(252).mean()
    ).astype(int)
)

# Volatility percentile rank (0-1):
df['vol_rank_30'] = (
    df['vol_roll_30'].rolling(window=252).rank(pct=True)
)

# Extreme high volatility regime (top 20%):
df['high_vol_regime'] = np.where(
    df['vol_rank_30'].isna(),
    np.nan,
    (df['vol_rank_30'] > 0.8).astype(int)
)

# Volatility expansion (increasing vs 5 days ago):
df['vol_expansion'] = np.where(
    df['vol_roll_30'].shift(5).isna(),
    np.nan,
    (
        df['vol_roll_30']
        > df['vol_roll_30'].shift(5)
    ).astype(int)
)

# Short-term vs medium-term volatility ratio:
df['vol_ratio_10_60'] = df['vol_roll_10'] / df['vol_roll_60']

print("Volatility regime features created successfully")
```

    Volatility regime features created successfully
    


```python
display(Markdown(f"""
## Volatility regime features

These features encode market risk state — whether volatility is elevated,
expanding, or historically extreme — giving models context that raw volatility
levels alone cannot provide.

**Features created:**

- `vol_above_avg` — whether 30-day rolling volatility is above its trailing
  1-year average
- `vol_rank_30` — current 30-day volatility expressed as a percentile rank
  within the past 252 trading days (0–1 scale)
- `high_vol_regime` — binary flag for periods where `vol_rank_30` exceeds 0.8,
  i.e. the top 20% historically
- `vol_expansion` — whether 30-day volatility is higher than it was 5 trading
  days earlier
- `vol_ratio_10_60` — ratio of 10-day to 60-day rolling volatility; values
  above 1 indicate short-term stress building relative to medium-term norms

Markets do not maintain constant risk. Volatility transitions between calm and
turbulent regimes — a pattern confirmed by the ARCH-LM test
(LM = {arch_lm:,.2f}, {arch_lm_nlags} lags,
p {'< 0.001' if arch_lm_p < 0.001 else f'= {arch_lm_p:.4f}'})
in the diagnostics notebook. These features operationalise that finding
by making regime state an explicit input rather than something a model must
infer entirely from returns.

Whether they improve volatility forecasting and regime classification is tested
in Notebooks 05 and 06.
"""))
```



## Volatility regime features

These features encode market risk state — whether volatility is elevated,
expanding, or historically extreme — giving models context that raw volatility
levels alone cannot provide.

**Features created:**

- `vol_above_avg` — whether 30-day rolling volatility is above its trailing
  1-year average
- `vol_rank_30` — current 30-day volatility expressed as a percentile rank
  within the past 252 trading days (0–1 scale)
- `high_vol_regime` — binary flag for periods where `vol_rank_30` exceeds 0.8,
  i.e. the top 20% historically
- `vol_expansion` — whether 30-day volatility is higher than it was 5 trading
  days earlier
- `vol_ratio_10_60` — ratio of 10-day to 60-day rolling volatility; values
  above 1 indicate short-term stress building relative to medium-term norms

Markets do not maintain constant risk. Volatility transitions between calm and
turbulent regimes — a pattern confirmed by the ARCH-LM test
(LM = 1,786.70, 10 lags,
p < 0.001)
in the diagnostics notebook. These features operationalise that finding
by making regime state an explicit input rather than something a model must
infer entirely from returns.

Whether they improve volatility forecasting and regime classification is tested
in Notebooks 05 and 06.



# Missing values handling


```python
# Check for NaN values in my Dataframe:
df.isna().sum().sort_values(ascending=False)
```




    vol_above_avg      280
    vol_rank_30        280
    high_vol_regime    280
    momentum_60         59
    vol_ratio_10_60     59
                      ... 
    month_6              0
    month_5              0
    month_12             0
    month_10             0
    month_11             0
    Length: 64, dtype: int64




```python
# Creating the dataframe for modelling
df_model = df.dropna().copy()
```


```python
# The final shape of the data 
print(f"Rows: {df_model.shape[0]:,}")
print(f"Columns: {df_model.shape[1]}")
df_model.info()
```

    Rows: 6,412
    Columns: 64
    <class 'pandas.DataFrame'>
    DatetimeIndex: 6412 entries, 2001-02-13 to 2026-08-14
    Data columns (total 64 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   Open                 6412 non-null   float64
     1   High                 6412 non-null   float64
     2   Low                  6412 non-null   float64
     3   Close                6412 non-null   float64
     4   Volume               6412 non-null   int64  
     5   log_returns          6412 non-null   float64
     6   return_lag_1         6412 non-null   float64
     7   return_lag_2         6412 non-null   float64
     8   return_lag_3         6412 non-null   float64
     9   return_lag_5         6412 non-null   float64
     10  close_pct_lag_1      6412 non-null   float64
     11  close_pct_lag_2      6412 non-null   float64
     12  range_lag_1          6412 non-null   float64
     13  volume_lag_1         6412 non-null   float64
     14  vol_roll_5           6412 non-null   float64
     15  return_mean_roll_5   6412 non-null   float64
     16  momentum_5           6412 non-null   float64
     17  vol_roll_10          6412 non-null   float64
     18  return_mean_roll_10  6412 non-null   float64
     19  momentum_10          6412 non-null   float64
     20  vol_roll_21          6412 non-null   float64
     21  return_mean_roll_21  6412 non-null   float64
     22  momentum_21          6412 non-null   float64
     23  vol_roll_30          6412 non-null   float64
     24  return_mean_roll_30  6412 non-null   float64
     25  momentum_30          6412 non-null   float64
     26  vol_roll_60          6412 non-null   float64
     27  return_mean_roll_60  6412 non-null   float64
     28  momentum_60          6412 non-null   float64
     29  rsi_14               6412 non-null   float64
     30  macd                 6412 non-null   float64
     31  macd_signal          6412 non-null   float64
     32  macd_hist            6412 non-null   float64
     33  bb_middle            6412 non-null   float64
     34  bb_std               6412 non-null   float64
     35  bb_upper             6412 non-null   float64
     36  bb_lower             6412 non-null   float64
     37  bb_width             6412 non-null   float64
     38  atr_14               6412 non-null   float64
     39  day_of_week          6412 non-null   int32  
     40  month                6412 non-null   int32  
     41  quarter              6412 non-null   int32  
     42  is_month_end         6412 non-null   int64  
     43  is_quarter_end       6412 non-null   int64  
     44  dow_1                6412 non-null   int64  
     45  dow_2                6412 non-null   int64  
     46  dow_3                6412 non-null   int64  
     47  dow_4                6412 non-null   int64  
     48  month_2              6412 non-null   int64  
     49  month_3              6412 non-null   int64  
     50  month_4              6412 non-null   int64  
     51  month_5              6412 non-null   int64  
     52  month_6              6412 non-null   int64  
     53  month_7              6412 non-null   int64  
     54  month_8              6412 non-null   int64  
     55  month_9              6412 non-null   int64  
     56  month_10             6412 non-null   int64  
     57  month_11             6412 non-null   int64  
     58  month_12             6412 non-null   int64  
     59  vol_above_avg        6412 non-null   float64
     60  vol_rank_30          6412 non-null   float64
     61  high_vol_regime      6412 non-null   float64
     62  vol_expansion        6412 non-null   float64
     63  vol_ratio_10_60      6412 non-null   float64
    dtypes: float64(43), int32(3), int64(18)
    memory usage: 3.1 MB
    


```python
# Check for infinite values
np.isinf(df.select_dtypes(include=np.number)).sum().sum()
```




    np.int64(0)




```python
# Save our data into a parquet dataframe to reserve data structure
df_model.to_parquet(
    "../data/sp500_features.parquet"
)
```

## ✅ Feature engineering summary

Five feature groups were constructed from the validated EDA output:

1. **Lag features** — recent return history, price changes, intraday range,
   and volume (lags 1–5 days)
2. **Rolling statistics** — rolling volatility, mean returns, and
   price-to-moving-average momentum across 5 to 60-day windows
3. **Technical indicators** — RSI, MACD, Bollinger Bands, ATR
4. **Calendar features** — day-of-week and month dummies, period-end flags
5. **Volatility regime features** — percentile ranks, high-volatility flags,
   expansion signals, short-to-medium-term volatility ratios

Each group was built in response to something observed in the diagnostics
notebook. Volatility clustering and conditional heteroskedasticity motivated
the regime and rolling features. Weak directional dependence in returns shaped
lag selection. Seasonal patterns in exploratory analysis justified the calendar
features.

The question now is whether these features improve forecasting accuracy,
volatility prediction, and regime classification. That is tested in the
modelling notebooks.
