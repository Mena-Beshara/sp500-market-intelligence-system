# S&P 500 data preparation and exploratory data analysis

This notebook acquires the raw S&P 500 index history, validates it against the
exchange trading calendar, transforms closing prices into log returns, and
establishes the statistical properties that govern every modelling decision in
the notebooks that follow.

All headline statistics produced here are written to `locked_metrics.json` and
read by key downstream. No statistic is typed by hand into prose in this
notebook. Every figure quoted in markdown is interpolated from a live variable,
every subsample window is defined once as a named constant, and every quantity
is rendered through a single format, so the same number cannot appear twice in
two different forms.


```python
# Import the needed libraries
import numpy as np
import pandas as pd
import yfinance as yf
# Setting Plotly backend plotting for pandas
pd.options.plotting.backend = 'plotly'
import plotly.io as pio
pio.templates.default = 'plotly_dark'
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
import pandas_market_calendars as mcal
import json
from pathlib import Path
from IPython.display import Markdown, display

# --- Paths -------------------------------------------------------------
# Created before any figure or dataset is written so a missing directory fails
# here rather than midway through the notebook.
FIGURES_DIR = Path('../reports/figures')
DATA_DIR = Path('../data')
METRICS_PATH = DATA_DIR / 'locked_metrics.json'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Acquisition thresholds --------------------------------------------
TICKER = '^GSPC'
SAMPLE_START = '2000-01-01'
MIN_SESSIONS_PER_YEAR = 240   # deliberately loose floor for the truncation check
MAX_STALENESS_DAYS = 7        # longest plausible gap including holiday weekends

# --- Analysis windows --------------------------------------------------
# Defined once and reused by every cell that computes a subsample statistic,
# shades a chart region, or names a period in prose. A chart therefore cannot
# shade a different span from the figure quoted beside it.
CALM_WINDOW = ('2013', '2019')
GFC_WINDOW = ('2008-09', '2009-05')
COVID_WINDOW = ('2020-02', '2020-05')

# --- Rolling windows ---------------------------------------------------
SHORT_VOL_WINDOW = 21    # one trading month, used on the returns overlay chart
ROLLING_VOL_WINDOW = 30  # standard risk-reporting window
ROLLING_VOL_COL = f'rolling_vol_{ROLLING_VOL_WINDOW}'

# --- Reporting precision -----------------------------------------------
P_VALUE_FLOOR = 1e-4  # below this, report a bound rather than a value
CONTEXT_SESSIONS = 5  # sessions shown either side of an anomaly
```


```python
# --- Formatting and window helpers ------------------------------------

def fmt_pvalue(p, floor=P_VALUE_FLOOR):
    """Render a p-value for prose.

    statsmodels returns exactly 0.0 for the ADF p-value when MacKinnon's
    approximate interpolation falls below its tabulated range. Printing a
    probability of zero overstates the result: the defensible statement is that
    the p-value lies below the resolution of the approximation. Every cell that
    quotes a p-value routes through this function, so the same value cannot be
    rendered one way in a chart and another way in prose.
    """
    if p < floor:
        return f'< {floor:g}'
    return f'{p:.6f}'


def window_bounds(window):
    """Convert a ('start', 'end') period pair into concrete timestamps.

    pandas string slicing treats a partial date as the whole period, so
    '2009-05' as a slice endpoint includes all of May. Chart shading needs
    explicit bounds, and deriving them from the same constant keeps the shaded
    region identical to the slice used for the statistic.
    """
    start = pd.Timestamp(window[0])
    if len(window[1]) == 4:
        end = pd.Timestamp(window[1]) + pd.offsets.YearEnd(0)
    else:
        end = pd.Timestamp(window[1]) + pd.offsets.MonthEnd(0)
    return start, end


def window_bounds_iso(window):
    """Window bounds as ISO date strings, for Plotly layout attributes.

    Plotly serialises layout attributes to JSON and does not convert
    pd.Timestamp. Trace coordinates are handled, because array-likes are
    converted to datetime64 on the way in, but scalar layout attributes such as
    a vrect boundary or an annotation anchor are not. Passing a Timestamp there
    raises TypeError: Type is not JSON serializable.
    """
    start, end = window_bounds(window)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def window_label(window):
    """Human-readable label for a window, derived rather than typed."""
    start, end = window_bounds(window)
    return f'{start.strftime("%B %Y")} to {end.strftime("%B %Y")}'


def window_vol_pct(series, window):
    """Daily volatility in percent over a named window."""
    return series.loc[window[0]:window[1]].std() * 100


print('Helpers defined.')
print(f'Calm window:  {window_label(CALM_WINDOW)}')
print(f'GFC window:   {window_label(GFC_WINDOW)}')
print(f'COVID window: {window_label(COVID_WINDOW)}')
```

    Helpers defined.
    Calm window:  January 2013 to December 2019
    GFC window:   September 2008 to May 2009
    COVID window: February 2020 to May 2020
    

## Load data


```python
# === DATA DOWNLOAD ===

# End date set dynamically so the pipeline always pulls the most recent
# available data. The system produces a weekly risk signal every Monday to
# inform that day's decision, and that signal must reflect the most recent
# finalised session.
#
# yf.download treats `end` as EXCLUSIVE. Setting end = today would make today's
# bar unreachable, which in turn would make the incomplete-session guard below
# unreachable. The boundary is therefore today + 1 day: today's bar is
# retrieved if it exists, and the guard decides whether to keep it.
#
# auto_adjust is passed explicitly. For a price index the adjustment is a
# no-operation, but the yfinance default changed between versions and it
# governs the returned column schema. Pinning it here removes a silent version
# dependency, and the no-op claim is verified further down rather than assumed.

today = pd.Timestamp('today').normalize()
requested_end = today + pd.Timedelta(days=1)

raw = yf.download(
    TICKER,
    start=SAMPLE_START,
    end=requested_end,
    auto_adjust=False,
)

# Postconditions on acquisition. yfinance can return an empty or truncated
# frame without raising: a bad symbol, a rate limit, or an endpoint change all
# surface as missing data rather than an exception. These checks convert a
# silent corruption into a loud failure at the point of origin.

if raw.empty:
    raise ValueError(
        'Download returned an empty DataFrame. Check connectivity, the ticker '
        'symbol, and whether Yahoo is rate limiting.'
    )

span_years = (raw.index[-1] - pd.Timestamp(SAMPLE_START)).days / 365.25

# The floor is derived from the REQUESTED span, never the delivered one.
# Computing it from raw.index[-1] makes the check self-defeating: a truncated
# download shrinks the expected count alongside the actual count, so a range
# ending ten years early would still pass. This floor is a coarse pre-check;
# the NYSE calendar comparison further down is the precise instrument, and it
# catches interior gaps this cannot.
requested_span_years = (today - pd.Timestamp(SAMPLE_START)).days / 365.25
session_floor = int(requested_span_years * MIN_SESSIONS_PER_YEAR)

if len(raw) < session_floor:
    raise ValueError(
        f'Download returned {len(raw):,} sessions; expected at least '
        f'{session_floor:,} for the requested {requested_span_years:.2f} year '
        'span. The range is likely truncated.'
    )

staleness_days = (today - raw.index[-1]).days
if staleness_days > MAX_STALENESS_DAYS:
    raise ValueError(
        f'Most recent session is {raw.index[-1].date()}, {staleness_days} days '
        'old. Data may be stale or the feed may be delayed.'
    )

# Log DELIVERED coverage, not the requested boundary. An earlier revision
# printed the requested end date, which reads as inclusive and did not match
# the last row in the frame.
price_series_start = raw.index[0]

print(f'Delivered coverage: {raw.index[0].date()} to {raw.index[-1].date()}')
print(f'Total sessions: {len(raw):,}')
print(f'Span: {span_years:.2f} years, {len(raw) / span_years:.1f} sessions per year')
print(f'Most recent session is {staleness_days} day(s) old')
```

    [*********************100%***********************]  1 of 1 completed

    Delivered coverage: 2000-01-03 to 2026-08-17
    Total sessions: 6,695
    Span: 26.63 years, 251.4 sessions per year
    Most recent session is 0 day(s) old
    

    
    


```python
# check our loaded data
raw
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Price</th>
      <th>Adj Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Ticker</th>
      <th>^GSPC</th>
      <th>^GSPC</th>
      <th>^GSPC</th>
      <th>^GSPC</th>
      <th>^GSPC</th>
      <th>^GSPC</th>
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
      <th>2000-01-03</th>
      <td>1455.219971</td>
      <td>1455.219971</td>
      <td>1478.000000</td>
      <td>1438.359985</td>
      <td>1469.250000</td>
      <td>931800000</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>1399.420044</td>
      <td>1399.420044</td>
      <td>1455.219971</td>
      <td>1397.430054</td>
      <td>1455.219971</td>
      <td>1009000000</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>1402.109985</td>
      <td>1402.109985</td>
      <td>1413.270020</td>
      <td>1377.680054</td>
      <td>1399.420044</td>
      <td>1085500000</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>1403.449951</td>
      <td>1403.449951</td>
      <td>1411.900024</td>
      <td>1392.099976</td>
      <td>1402.109985</td>
      <td>1092300000</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>1441.469971</td>
      <td>1441.469971</td>
      <td>1441.469971</td>
      <td>1400.729980</td>
      <td>1403.449951</td>
      <td>1225200000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2026-08-11</th>
      <td>7728.200195</td>
      <td>7728.200195</td>
      <td>7767.509766</td>
      <td>7717.250000</td>
      <td>7767.509766</td>
      <td>4739500000</td>
    </tr>
    <tr>
      <th>2026-08-12</th>
      <td>7748.500000</td>
      <td>7748.500000</td>
      <td>7766.009766</td>
      <td>7737.950195</td>
      <td>7765.459961</td>
      <td>4574170000</td>
    </tr>
    <tr>
      <th>2026-08-13</th>
      <td>7798.990234</td>
      <td>7798.990234</td>
      <td>7816.700195</td>
      <td>7763.180176</td>
      <td>7763.180176</td>
      <td>4833230000</td>
    </tr>
    <tr>
      <th>2026-08-14</th>
      <td>7785.759766</td>
      <td>7785.759766</td>
      <td>7810.009766</td>
      <td>7776.310059</td>
      <td>7806.600098</td>
      <td>4159900000</td>
    </tr>
    <tr>
      <th>2026-08-17</th>
      <td>7757.049805</td>
      <td>7757.049805</td>
      <td>7790.680176</td>
      <td>7751.830078</td>
      <td>7790.680176</td>
      <td>1400190000</td>
    </tr>
  </tbody>
</table>
<p>6695 rows × 6 columns</p>
</div>



## Structural validation


```python
# Index integrity. Explicit raises rather than assert statements: assert is
# stripped when Python runs under -O, so a validation check written as an
# assertion is a check that can be compiled out of existence. This logic is
# destined for a src/ validation module where that matters.

# yfinance occasionally returns duplicate dates.
duplicate_count = int(raw.index.duplicated().sum())
if duplicate_count > 0:
    offenders = raw.index[raw.index.duplicated(keep=False)].unique()
    raise ValueError(
        f'Found {duplicate_count} duplicate date(s) in the index: '
        f'{[str(d.date()) for d in offenders]}'
    )

# Uniqueness does not imply ordering. An index can be unique and still out of
# chronological sequence, which corrupts every lagged operation downstream.
if not raw.index.is_monotonic_increasing:
    raise ValueError('Index is not monotonically increasing.')

print(f'Duplicate dates: {duplicate_count}')
print(f'Index monotonic increasing: {raw.index.is_monotonic_increasing}')
print(f'Index type: {type(raw.index).__name__}')
```

    Duplicate dates: 0
    Index monotonic increasing: True
    Index type: DatetimeIndex
    


```python
# Calendar completeness against the NYSE schedule.
#
# The sessions-per-year arithmetic printed above is reassuring in aggregate but
# cannot rule out scattered missing sessions. This compares the index against
# the exchange schedule in both directions: sessions the exchange held that are
# absent from the data, and dates in the data on which the exchange was closed.
#
# This check runs BEFORE the zero-volume row is removed. Running it afterwards
# would flag that deliberate exclusion as a calendar gap.

nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(
    start_date=raw.index[0].strftime('%Y-%m-%d'),
    end_date=raw.index[-1].strftime('%Y-%m-%d'),
)
expected_sessions = pd.DatetimeIndex(schedule.index).normalize()
actual_sessions = raw.index.normalize()

missing_sessions = expected_sessions.difference(actual_sessions)
unexpected_sessions = actual_sessions.difference(expected_sessions)
calendar_complete = len(missing_sessions) == 0 and len(unexpected_sessions) == 0

print(f'NYSE sessions expected: {len(expected_sessions):,}')
print(f'Sessions in dataset:    {len(actual_sessions):,}')
print(f'Missing from dataset:   {len(missing_sessions)}')
print(f'Not on NYSE calendar:   {len(unexpected_sessions)}')

if len(missing_sessions) > 0:
    print(f'  Missing: {[str(d.date()) for d in missing_sessions[:10]]}')
if len(unexpected_sessions) > 0:
    print(f'  Unexpected: {[str(d.date()) for d in unexpected_sessions[:10]]}')
```

    NYSE sessions expected: 6,695
    Sessions in dataset:    6,695
    Missing from dataset:   0
    Not on NYSE calendar:   0
    


```python
display(Markdown(f"""
### Calendar completeness

The dataset index was compared against the New York Stock Exchange trading
schedule across the full sample, in both directions.

- NYSE sessions expected over the range: **{len(expected_sessions):,}**
- Sessions present in the dataset: **{len(actual_sessions):,}**
- Exchange sessions absent from the dataset: **{len(missing_sessions)}**
- Dataset dates on which the exchange was closed: **{len(unexpected_sessions)}**

{'Calendar coverage is complete. Every session the exchange held is present, and the dataset contains no dates outside the exchange schedule.' if calendar_complete else 'Calendar coverage is incomplete. The discrepancies listed above require investigation before modelling.'}

This is a stronger claim than a session count supports. Dividing
{len(raw):,} sessions by a {span_years:.2f} year span gives
{len(raw) / span_years:.1f} sessions per year against a benchmark near 252,
which is consistent with complete coverage but cannot distinguish complete
coverage from a handful of scattered omissions. The set comparison can.
"""))
```



### Calendar completeness

The dataset index was compared against the New York Stock Exchange trading
schedule across the full sample, in both directions.

- NYSE sessions expected over the range: **6,695**
- Sessions present in the dataset: **6,695**
- Exchange sessions absent from the dataset: **0**
- Dataset dates on which the exchange was closed: **0**

Calendar coverage is complete. Every session the exchange held is present, and the dataset contains no dates outside the exchange schedule.

This is a stronger claim than a session count supports. Dividing
6,695 sessions by a 26.63 year span gives
251.4 sessions per year against a benchmark near 252,
which is consistent with complete coverage but cannot distinguish complete
coverage from a handful of scattered omissions. The set comparison can.



### Understanding OHLCV market data

Each row in the dataset represents a single trading day in the market.

**Open.** The first recorded price at the start of the trading session, 9:30 AM
EST. It often reflects overnight news, earnings, macroeconomic developments,
and shifts in investor sentiment while markets were closed.

**High and low.** The maximum and minimum prices reached during the session.
Together they define the intraday range, which provides insight into market
volatility and uncertainty. Larger ranges typically indicate strong reactions
to new information or increased trading activity.

**Close.** The final price recorded at the end of the trading session, 4:00 PM
EST. It represents the market's final consensus on valuation and is the primary
variable used in financial analysis and modelling. Most time series models are
built on closing prices or on returns derived from them.

**Volume.** The level of trading activity during the day. For an index such as
the S&P 500 volume is not directly observed at the index level; it is derived
or supplied by the data vendor, which makes it less reliable than the price
fields.

In this notebook volume serves one purpose: a data integrity signal. A reported
volume of zero on a session the exchange confirmed as open identifies a corrupt
record, and that is exactly the check performed below. Volume enters the
modelling pipeline once, as `volume_lag_1` in Notebook 03, where a one-day lag
supplies liquidity context alongside the return and range lags. It is never
used as a price or return input, and no volatility estimate in this project
depends on it.

## Data inspection


```python
# Check the dataset shape
raw.shape
```




    (6695, 6)




```python
# Check for missing values
raw.isnull().sum()
```




    Price      Ticker
    Adj Close  ^GSPC     0
    Close      ^GSPC     0
    High       ^GSPC     0
    Low        ^GSPC     0
    Open       ^GSPC     0
    Volume     ^GSPC     0
    dtype: int64




```python
# Check data description
summary = raw.describe()
summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Price</th>
      <th>Adj Close</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Ticker</th>
      <th>^GSPC</th>
      <th>^GSPC</th>
      <th>^GSPC</th>
      <th>^GSPC</th>
      <th>^GSPC</th>
      <th>^GSPC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6695.000000</td>
      <td>6695.000000</td>
      <td>6695.000000</td>
      <td>6695.000000</td>
      <td>6695.000000</td>
      <td>6.695000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2382.841487</td>
      <td>2382.841487</td>
      <td>2395.921461</td>
      <td>2367.982472</td>
      <td>2382.445149</td>
      <td>3.467184e+09</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1616.168927</td>
      <td>1616.168927</td>
      <td>1623.209929</td>
      <td>1607.862675</td>
      <td>1615.745204</td>
      <td>1.534724e+09</td>
    </tr>
    <tr>
      <th>min</th>
      <td>676.530029</td>
      <td>676.530029</td>
      <td>695.270020</td>
      <td>666.789978</td>
      <td>679.280029</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1215.655029</td>
      <td>1215.655029</td>
      <td>1223.040039</td>
      <td>1209.144958</td>
      <td>1215.505005</td>
      <td>2.366070e+09</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1585.160034</td>
      <td>1585.160034</td>
      <td>1592.640015</td>
      <td>1577.560059</td>
      <td>1582.770020</td>
      <td>3.560750e+09</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2995.905029</td>
      <td>2995.905029</td>
      <td>3006.770020</td>
      <td>2980.400024</td>
      <td>2996.660034</td>
      <td>4.325015e+09</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7798.990234</td>
      <td>7798.990234</td>
      <td>7816.700195</td>
      <td>7776.310059</td>
      <td>7806.600098</td>
      <td>1.145623e+10</td>
    </tr>
  </tbody>
</table>
</div>




```python
display(Markdown(
    f"The dataset contains **{len(raw):,} daily observations** spanning "
    f"**{span_years:.2f} years**, providing a sufficiently large sample for "
    f"statistical analysis and modelling."
))
```


The dataset contains **6,695 daily observations** spanning **26.63 years**, providing a sufficiently large sample for statistical analysis and modelling.



```python
# Inspect the current column structure
raw.columns
```




    MultiIndex([('Adj Close', '^GSPC'),
                (    'Close', '^GSPC'),
                (     'High', '^GSPC'),
                (      'Low', '^GSPC'),
                (     'Open', '^GSPC'),
                (   'Volume', '^GSPC')],
               names=['Price', 'Ticker'])




```python
# Verify the auto_adjust claim rather than assert it.
#
# auto_adjust=False returns Adj Close alongside Close. For a price index there
# are no dividends or splits to adjust for, so the two should be identical. If
# they ever diverge, the ticker is not behaving as a price index and the
# assumption behind dropping Adj Close is wrong.

if isinstance(raw.columns, pd.MultiIndex):
    has_adj = 'Adj Close' in raw.columns.get_level_values(0)
    adj_close_col = raw[('Adj Close', TICKER)] if has_adj else None
    close_col = raw[('Close', TICKER)]
else:
    has_adj = 'Adj Close' in raw.columns
    adj_close_col = raw['Adj Close'] if has_adj else None
    close_col = raw['Close']

if adj_close_col is not None:
    max_adj_gap = float((adj_close_col - close_col).abs().max())
    adj_close_identical = max_adj_gap == 0.0
    print(f'Largest absolute Adj Close minus Close difference: {max_adj_gap}')
    print(f'Adj Close identical to Close: {adj_close_identical}')
    if not adj_close_identical:
        raise ValueError(
            'Adj Close differs from Close. This ticker is expected to carry no '
            'dividend or split adjustment; investigate before dropping the column.'
        )
else:
    max_adj_gap = None
    adj_close_identical = None
    print('No Adj Close column returned; auto_adjust may have been overridden.')
```

    Largest absolute Adj Close minus Close difference: 0.0
    Adj Close identical to Close: True
    


```python
# Flatten multi-level columns
#
# yfinance returns a MultiIndex pairing each field with the ticker symbol.
# That structure is useful across multiple assets and is pure overhead for a
# single index. Selecting the five fields by name also discards Adj Close,
# which the check above confirmed is identical to Close for this ticker.

if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

print('Columns after flattening:', raw.columns.tolist())
```

    Columns after flattening: ['Open', 'High', 'Low', 'Close', 'Volume']
    


```python
raw.head()
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
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-03</th>
      <td>1469.250000</td>
      <td>1478.000000</td>
      <td>1438.359985</td>
      <td>1455.219971</td>
      <td>931800000</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>1455.219971</td>
      <td>1455.219971</td>
      <td>1397.430054</td>
      <td>1399.420044</td>
      <td>1009000000</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>1399.420044</td>
      <td>1413.270020</td>
      <td>1377.680054</td>
      <td>1402.109985</td>
      <td>1085500000</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>1402.109985</td>
      <td>1411.900024</td>
      <td>1392.099976</td>
      <td>1403.449951</td>
      <td>1092300000</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>1403.449951</td>
      <td>1441.469971</td>
      <td>1400.729980</td>
      <td>1441.469971</td>
      <td>1225200000</td>
    </tr>
  </tbody>
</table>
</div>




```python
display(Markdown(f"""
### Column structure adjustment: flattening multi-level headers

The dataset arrives with multi-level column headers, because yfinance pairs
each field with the ticker symbol. That format is useful when working with
multiple assets and introduces unnecessary complexity in a single-index
context: it complicates column selection and referencing, and it raises the
risk of error in feature engineering, transformations, and model input
preparation.

The columns were flattened to a single level and the five fields selected
explicitly by name. Standardising the schema at acquisition keeps column access
simple and makes it independent of the library version that produced it.

`Adj Close` was dropped rather than retained. The largest absolute difference
between `Adj Close` and `Close` across all {len(raw):,} sessions is
**{max_adj_gap}**, confirming that no dividend or split adjustment applies to
this series. That is the expected behaviour for a price index, and it is
verified here rather than assumed.
"""))
```



### Column structure adjustment: flattening multi-level headers

The dataset arrives with multi-level column headers, because yfinance pairs
each field with the ticker symbol. That format is useful when working with
multiple assets and introduces unnecessary complexity in a single-index
context: it complicates column selection and referencing, and it raises the
risk of error in feature engineering, transformations, and model input
preparation.

The columns were flattened to a single level and the five fields selected
explicitly by name. Standardising the schema at acquisition keeps column access
simple and makes it independent of the library version that produced it.

`Adj Close` was dropped rather than retained. The largest absolute difference
between `Adj Close` and `Close` across all 6,695 sessions is
**0.0**, confirming that no dividend or split adjustment applies to
this series. That is the expected behaviour for a price index, and it is
verified here rather than assumed.




```python
display(Markdown(
    f"The minimum value of the **Volume** column is "
    f"**{raw['Volume'].min():,.0f}**, which is unexpected for a confirmed "
    f"trading day.\n\n"
    f"This warrants further investigation to assess whether it represents a "
    f"data integrity issue or a legitimate market condition."
))
```


The minimum value of the **Volume** column is **0**, which is unexpected for a confirmed trading day.

This warrants further investigation to assess whether it represents a data integrity issue or a legitimate market condition.


## Zero-volume observation


```python
zero_vol = raw[raw['Volume'] == 0].copy()
sessions_before = len(raw)

print(f'Zero-volume observations detected: {len(zero_vol)}')
if len(zero_vol) > 0:
    print(f'Affected date(s): {[str(d.date()) for d in zero_vol.index]}')
```

    Zero-volume observations detected: 1
    Affected date(s): ['2023-05-24']
    


```python
# This pipeline is calibrated for a single known anomaly.
# If yfinance revises historical data or a new anomaly appears,
# this raises immediately rather than silently removing multiple
# observations.

if len(zero_vol) > 1:
    raise ValueError(
        f'Expected at most one zero-volume observation; found {len(zero_vol)}. '
        'Inspect zero_vol before proceeding.'
    )
```


```python
# Timedelta does not guarantee a fixed number of trading sessions — weekends
# can shorten the window. iloc gives exactly CONTEXT_SESSIONS observations on
# each side.

if len(zero_vol) > 0:
    idx = zero_vol.index[0]
    loc = raw.index.get_loc(idx)

    # Clamp to valid index bounds
    window = raw.iloc[
        max(0, loc - CONTEXT_SESSIONS): min(len(raw), loc + CONTEXT_SESSIONS + 1)
    ]

    print(f'{CONTEXT_SESSIONS} trading sessions before and after the '
          'zero-volume observation')
    display(window[['Close', 'Volume']])
```

    5 trading sessions before and after the zero-volume observation
    


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
      <th>Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-05-17</th>
      <td>4158.770020</td>
      <td>4039080000</td>
    </tr>
    <tr>
      <th>2023-05-18</th>
      <td>4198.049805</td>
      <td>3980500000</td>
    </tr>
    <tr>
      <th>2023-05-19</th>
      <td>4191.979980</td>
      <td>4041900000</td>
    </tr>
    <tr>
      <th>2023-05-22</th>
      <td>4192.629883</td>
      <td>3728520000</td>
    </tr>
    <tr>
      <th>2023-05-23</th>
      <td>4145.580078</td>
      <td>4155320000</td>
    </tr>
    <tr>
      <th>2023-05-24</th>
      <td>4115.240234</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2023-05-25</th>
      <td>4151.279785</td>
      <td>4147760000</td>
    </tr>
    <tr>
      <th>2023-05-26</th>
      <td>4205.450195</td>
      <td>3715460000</td>
    </tr>
    <tr>
      <th>2023-05-30</th>
      <td>4205.520020</td>
      <td>4228510000</td>
    </tr>
    <tr>
      <th>2023-05-31</th>
      <td>4179.830078</td>
      <td>5980670000</td>
    </tr>
    <tr>
      <th>2023-06-01</th>
      <td>4221.020020</td>
      <td>4391860000</td>
    </tr>
  </tbody>
</table>
</div>



```python
# holidays.US() tests against the federal holiday calendar.
# The NYSE does not follow the federal calendar exactly —
# Good Friday, Juneteenth (since 2022), and national days of
# mourning are observed by the exchange but not federal statute.
# pandas_market_calendars uses the NYSE schedule directly and
# is more defensible for a market data pipeline.

if len(zero_vol) > 0:
    zero_date = zero_vol.index[0]
    zero_close = float(zero_vol['Close'].iloc[0])

    day_schedule = nyse.schedule(
        start_date=zero_date.strftime('%Y-%m-%d'),
        end_date=zero_date.strftime('%Y-%m-%d'),
    )

    if day_schedule.empty:
        market_status = 'NYSE was closed on this date.'
        holiday_note = 'falls on a NYSE non-trading day'
    else:
        market_status = 'NYSE was open on this date — expected trading session.'
        holiday_note = (
            'does not fall on a NYSE holiday — treated as a data integrity error'
        )

    print(f'{zero_date.date()} | {market_status}')
    print(f'Reported close on that session: {zero_close:,.2f}')
```

    2023-05-24 | NYSE was open on this date — expected trading session.
    Reported close on that session: 4,115.24
    


```python
# The chart is built from raw — the full dataset before the
# anomaly is removed. This ensures the zero-volume observation
# is visible on the series. After raw is filtered below, the
# point no longer exists in the index.

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=raw.index,
        y=raw['Volume'],
        mode='lines',
        name='Volume',
        line=dict(width=1, color='#5DCAA5')
    )
)

if len(zero_vol) > 0:
    anomaly_date_str = str(zero_date.date())  # Plotly cannot serialise pd.Timestamp directly
    anomaly_volume = zero_vol.loc[zero_date, 'Volume']  # 0 — from zero_vol, not raw

    fig.add_trace(
        go.Scatter(
            x=[anomaly_date_str],
            y=[anomaly_volume],
            mode='markers',
            name='Zero-volume observation',
            marker=dict(size=12, color='#E24B4A', symbol='x')
        )
    )

    fig.add_annotation(
        x=anomaly_date_str,
        y=anomaly_volume,
        text=(
            f'Zero volume: {zero_date.date()}'
            '<br>Expected NYSE session'
            '<br>Removed before modelling'
        ),
        showarrow=True,
        arrowhead=2,
        ax=130,
        ay=-80,
        font=dict(size=11),
        bgcolor='rgba(30,30,30,0.75)',
        bordercolor='#E24B4A',
        borderwidth=1
    )

fig.update_layout(
    title='S&P 500 volume — anomaly detection (pre-removal)',
    xaxis_title='Date',
    yaxis_title='Volume',
    height=500,
    width=1400,
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)

fig.show()

fig.write_image(
    FIGURES_DIR / 'sp500_volume_anomaly.png',
    width=1200,
    height=627,
    scale=2
)
```



## Zero-volume investigation


```python
# The row is dropped in full rather than having Volume set to NaN.
#
# The alternative was considered: Close on the affected session is a plausible
# value, and preserving the price row would keep the return series continuous.
# Dropping the row instead means the return dated on the following session is
# computed against the session two days prior, so one observation in the series
# is a two-day return carrying a one-day label.
#
# The row is dropped anyway. Volume is the integrity signal for this dataset,
# and a record whose volume field is known corrupt is a record whose other
# fields carry no guarantee. Retaining a price because it looks reasonable
# substitutes judgement for verification. One spliced return moves no aggregate
# statistic materially, whereas a silently corrupt row propagating through
# feature engineering and model estimation is the failure mode this pipeline
# exists to prevent.

raw = raw[raw['Volume'] > 0]
sessions_after = len(raw)

display(Markdown(f"""
## Data integrity: zero-volume observation

During data validation, {len(zero_vol)} zero-volume observation(s) were
identified in the downloaded dataset.

The S&P 500 index is published for every regular NYSE trading session. A zero
value in the downloaded Volume field is inconsistent with normal market
activity and is treated as a data integrity issue, not a valid market
observation.

The affected observation occurred on **{zero_date.date()}**, with a reported
close of **{zero_close:,.2f}**. Calendar verification against the NYSE schedule
confirmed that it {holiday_note}.

The observation was removed before calculating returns, rolling statistics, or
fitting any forecasting models. A single erroneous record propagating through
feature engineering and model estimation could bias volatility estimates and
invalidate downstream diagnostics.

The row was removed in full rather than retaining the price and nulling the
volume field. The consequence is one spliced return in the series, dated the
following session and spanning two calendar sessions. That cost is accepted:
volume is the integrity signal for this dataset, and the remaining fields on a
record with a known corrupt field carry no guarantee.

---

Sessions before cleaning: **{sessions_before:,}**
Observations removed: **{len(zero_vol)}**
Sessions after cleaning: **{sessions_after:,}**
Range after cleaning: **{raw.index[0].date()} to {raw.index[-1].date()}**
"""))
```



## Data integrity: zero-volume observation

During data validation, 1 zero-volume observation(s) were
identified in the downloaded dataset.

The S&P 500 index is published for every regular NYSE trading session. A zero
value in the downloaded Volume field is inconsistent with normal market
activity and is treated as a data integrity issue, not a valid market
observation.

The affected observation occurred on **2023-05-24**, with a reported
close of **4,115.24**. Calendar verification against the NYSE schedule
confirmed that it does not fall on a NYSE holiday — treated as a data integrity error.

The observation was removed before calculating returns, rolling statistics, or
fitting any forecasting models. A single erroneous record propagating through
feature engineering and model estimation could bias volatility estimates and
invalidate downstream diagnostics.

The row was removed in full rather than retaining the price and nulling the
volume field. The consequence is one spliced return in the series, dated the
following session and spanning two calendar sessions. That cost is accepted:
volume is the integrity signal for this dataset, and the remaining fields on a
record with a known corrupt field carry no guarantee.

---

Sessions before cleaning: **6,695**
Observations removed: **1**
Sessions after cleaning: **6,694**
Range after cleaning: **2000-01-03 to 2026-08-17**




```python
# Confirm no zero-volume rows remain
raw[raw['Volume'] == 0]
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
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# === DATA INTEGRITY: INCOMPLETE SESSION HANDLING ===

# The live signal requirement governs this logic. The system produces a weekly
# risk signal every Monday to inform that day's decision, and that signal must
# reflect Friday's complete close — the most recent finalised session.
#
# An unconditional drop of the last row would silently remove Friday's data
# when the pipeline runs over the weekend or on Monday morning, generating a
# signal from Thursday's figures with no indication anything was wrong.
#
# The guard therefore drops the last row only when it represents today's
# session. Today's bar is never trusted: nothing in the returned data
# distinguishes a completed session from a partial one, so the T-1 discipline
# is enforced on the date rather than on an inference about market hours.
#
# The guard is reachable because the download boundary is today + 1 day. Under
# an end boundary of today, exclusivity meant today's bar could never appear
# and this branch could never execute. Reachable is not the same as exercised:
# the branch fires only on a run where Yahoo has already published a bar for
# the current date, so a run before the session opens reports no drop.

if raw.index[-1] >= today:
    dropped_session = raw.index[-1].date()
    raw = raw.iloc[:-1]
    incomplete_session_dropped = True
    print(f'Incomplete session detected and removed: {dropped_session}')
else:
    incomplete_session_dropped = False
    print(f'Last confirmed session retained: {raw.index[-1].date()}')
    print(f"Today is {today.date()} and no bar for it was returned, so the "
          'guard did not fire on this run.')
```

    Incomplete session detected and removed: 2026-08-17
    


```python
# Confirm the last row
raw.tail()
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
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2026-08-10</th>
      <td>7751.740234</td>
      <td>7773.759766</td>
      <td>7743.109863</td>
      <td>7753.109863</td>
      <td>4879540000</td>
    </tr>
    <tr>
      <th>2026-08-11</th>
      <td>7767.509766</td>
      <td>7767.509766</td>
      <td>7717.250000</td>
      <td>7728.200195</td>
      <td>4739500000</td>
    </tr>
    <tr>
      <th>2026-08-12</th>
      <td>7765.459961</td>
      <td>7766.009766</td>
      <td>7737.950195</td>
      <td>7748.500000</td>
      <td>4574170000</td>
    </tr>
    <tr>
      <th>2026-08-13</th>
      <td>7763.180176</td>
      <td>7816.700195</td>
      <td>7763.180176</td>
      <td>7798.990234</td>
      <td>4833230000</td>
    </tr>
    <tr>
      <th>2026-08-14</th>
      <td>7806.600098</td>
      <td>7810.009766</td>
      <td>7776.310059</td>
      <td>7785.759766</td>
      <td>4159900000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualise close price over time
fig = raw['Close'].plot(
    title=f'S&P 500 close price ({raw.index[0].year} to {raw.index[-1].year})'
)
fig.update_layout(
    showlegend=False,
    yaxis_title='S&P 500',
    xaxis_title='Date'
)
fig.show()
fig.write_image(
    FIGURES_DIR / 'sp500_close_price.png',
    width=1400,
    height=700,
    scale=2
)
```



## Data stationarity assumptions

To assess suitability for financial time series modelling, raw closing prices
are transformed into log returns.

Raw prices are typically non-stationary, exhibiting trends and compounding
effects that violate the assumptions of many statistical models. Transforming
prices into log returns removes these effects and produces a series with more
stable statistical properties. Log returns are widely used in quantitative
finance because they are additive through time and suited to modelling
relative price changes.

### Achieving stationarity

| Property | Issue | Fix | Test |
| --- | --- | --- | --- |
| Stationarity in variance | Changing volatility | Log transform / GARCH | ARCH |
| Stationarity in mean | Trend / drift | Differencing | ADF |


```python
# Create a new column and calculate the daily log returns
raw['log_returns'] = np.log(raw['Close'] / raw['Close'].shift(1))
```


```python
# Double check our results
raw[['Close', 'log_returns']].head(10)
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
      <th>Close</th>
      <th>log_returns</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-03</th>
      <td>1455.219971</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>1399.420044</td>
      <td>-0.039099</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>1402.109985</td>
      <td>0.001920</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>1403.449951</td>
      <td>0.000955</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>1441.469971</td>
      <td>0.026730</td>
    </tr>
    <tr>
      <th>2000-01-10</th>
      <td>1457.599976</td>
      <td>0.011128</td>
    </tr>
    <tr>
      <th>2000-01-11</th>
      <td>1438.560059</td>
      <td>-0.013149</td>
    </tr>
    <tr>
      <th>2000-01-12</th>
      <td>1432.250000</td>
      <td>-0.004396</td>
    </tr>
    <tr>
      <th>2000-01-13</th>
      <td>1449.680054</td>
      <td>0.012096</td>
    </tr>
    <tr>
      <th>2000-01-14</th>
      <td>1465.150024</td>
      <td>0.010615</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop the first row: a daily return cannot be computed for the first session.
# Both start dates are retained as variables. Conflating the price series start
# with the return series start is a recurring source of confusion in write-ups,
# so each is recorded and quoted explicitly.

raw = raw.dropna(subset=['log_returns'])
return_series_start = raw.index[0]

print(f'Price series began:   {price_series_start.date()}')
print(f'Return series begins: {return_series_start.date()}')
print(f'Remaining NaN in log_returns: {raw["log_returns"].isna().sum()}')
```

    Price series began:   2000-01-03
    Return series begins: 2000-01-04
    Remaining NaN in log_returns: 0
    


```python
raw['log_returns'].describe()
```




    count    6692.000000
    mean        0.000251
    std         0.012144
    min        -0.127652
    25%        -0.004712
    50%         0.000639
    75%         0.005885
    max         0.109572
    Name: log_returns, dtype: float64




```python
daily_vol = raw['log_returns'].std() * 100
mean_log_return = raw['log_returns'].mean()

display(Markdown(f"""
The mean of log returns is **{mean_log_return:.6f}**, close to zero and
consistent with empirical findings in financial markets. That alone does not
imply predictability, and it does not confirm stationarity. A formal test is
required, and the Augmented Dickey-Fuller test is applied below.

The standard deviation of log returns represents daily volatility, the typical
magnitude of price fluctuations. In this dataset it is approximately
**{daily_vol:.2f}% per day**, consistent with historical equity market
behaviour.
"""))
```



The mean of log returns is **0.000251**, close to zero and
consistent with empirical findings in financial markets. That alone does not
imply predictability, and it does not confirm stationarity. A formal test is
required, and the Augmented Dickey-Fuller test is applied below.

The standard deviation of log returns represents daily volatility, the typical
magnitude of price fluctuations. In this dataset it is approximately
**1.21% per day**, consistent with historical equity market
behaviour.




```python
# Identify the dates of extreme moves
raw['log_returns'].nsmallest(5)
```




    Date
    2020-03-16   -0.127652
    2020-03-12   -0.099945
    2008-10-15   -0.094695
    2008-12-01   -0.093537
    2008-09-29   -0.092190
    Name: log_returns, dtype: float64




```python
raw['log_returns'].nlargest(5)
```




    Date
    2008-10-13    0.109572
    2008-10-28    0.102457
    2025-04-09    0.090895
    2020-03-24    0.089683
    2020-03-13    0.088808
    Name: log_returns, dtype: float64




```python
extreme_loss = raw['log_returns'].min()
extreme_loss_date = raw['log_returns'].idxmin()
extreme_gain = raw['log_returns'].max()
extreme_gain_date = raw['log_returns'].idxmax()

# Log returns and simple returns diverge at large magnitudes. Both are stated
# so the units of any quoted figure are unambiguous.
extreme_loss_simple = np.expm1(extreme_loss)
extreme_gain_simple = np.expm1(extreme_gain)

# Express the extremes in standard deviations of the unconditional distribution,
# derived rather than described loosely as "multiple standard deviations".
return_sd = raw['log_returns'].std()
extreme_loss_sigma = extreme_loss / return_sd
extreme_gain_sigma = extreme_gain / return_sd

display(Markdown(f"""
### Extreme market movements

The largest positive and negative daily returns were examined to characterise
tail behaviour.

The largest single-day decline is a log return of **{extreme_loss:.6f}** on
**{extreme_loss_date.date()}**, equivalent to a simple return of
**{extreme_loss_simple:.2%}** and **{extreme_loss_sigma:.1f}** unconditional
standard deviations. The largest single-day gain is a log return of
**{extreme_gain:.6f}** on **{extreme_gain_date.date()}**, equivalent to a
simple return of **{extreme_gain_simple:.2%}** and
**{extreme_gain_sigma:.1f}** standard deviations.

Large moves of both signs occupy the same short windows rather than arriving
independently, which is the first visible evidence of volatility clustering.

Under a normal distribution with this sample's standard deviation, a move of
{abs(extreme_loss_sigma):.1f} standard deviations carries a probability far
below one in a billion per session. Observing one inside {len(raw):,} sessions
is direct evidence that returns are not normally distributed, tested formally
in Notebook 02 and modelled through Student's t innovations in Notebook 05.
"""))
```



### Extreme market movements

The largest positive and negative daily returns were examined to characterise
tail behaviour.

The largest single-day decline is a log return of **-0.127652** on
**2020-03-16**, equivalent to a simple return of
**-11.98%** and **-10.5** unconditional
standard deviations. The largest single-day gain is a log return of
**0.109572** on **2008-10-13**, equivalent to a
simple return of **11.58%** and
**9.0** standard deviations.

Large moves of both signs occupy the same short windows rather than arriving
independently, which is the first visible evidence of volatility clustering.

Under a normal distribution with this sample's standard deviation, a move of
10.5 standard deviations carries a probability far
below one in a billion per session. Observing one inside 6,692 sessions
is direct evidence that returns are not normally distributed, tested formally
in Notebook 02 and modelled through Student's t innovations in Notebook 05.




```python
# Create figure with secondary y-axis for rolling volatility
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(
        x=raw.index,
        y=raw['log_returns'],
        name='Daily log returns',
        line=dict(color='#1E88E5', width=1.2),
        opacity=0.85
    ),
    secondary_y=False
)

# The window constant drives both the calculation and the trace label, so the
# legend cannot describe a different window from the one plotted.
short_rolling_vol = (
    raw['log_returns'].rolling(window=SHORT_VOL_WINDOW).std() * 100
)

fig.add_trace(
    go.Scatter(
        x=raw.index,
        y=short_rolling_vol,
        name=f'{SHORT_VOL_WINDOW}-day rolling volatility (%)',
        line=dict(color='#D32F2F', width=2.5),
        opacity=0.9
    ),
    secondary_y=True
)

fig.update_layout(
    title=(
        f'S&P 500 daily log returns vs {SHORT_VOL_WINDOW}-day rolling volatility'
    ),
    height=720,
    width=1450,
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    margin=dict(l=50, r=50, t=80, b=50)
)

fig.update_yaxes(title_text='Daily log return', secondary_y=False, gridcolor='rgba(255,255,255,0.08)')
fig.update_yaxes(title_text='Rolling volatility (%)', secondary_y=True, gridcolor='rgba(255,255,255,0.08)')

# Shaded regions derive from the same constants as the subsample statistics
# below, so a shaded span cannot disagree with a quoted figure.
for stress_window in (GFC_WINDOW, COVID_WINDOW):
    w_start, w_end = window_bounds_iso(stress_window)
    fig.add_vrect(
        x0=w_start,
        x1=w_end,
        fillcolor='red',
        opacity=0.1,
        layer='below',
        line_width=0
    )

fig.show()

fig.write_image(
    FIGURES_DIR / 'sp500_log_returns_with_vol.png',
    width=1450,
    height=720,
    scale=2.5
)
```




```python
# Subsample volatilities computed once here and reused wherever quoted.
# Variable names deliberately avoid embedding dates, so a name cannot drift out
# of agreement with the window constant that produced it.

vol_calm = window_vol_pct(raw['log_returns'], CALM_WINDOW)
vol_gfc = window_vol_pct(raw['log_returns'], GFC_WINDOW)
vol_covid = window_vol_pct(raw['log_returns'], COVID_WINDOW)
stress_ratio = vol_covid / vol_calm

display(Markdown(f"""
### Volatility clustering

The log return series exhibits volatility clustering: periods of high
volatility are followed by further high volatility, and periods of relative
calm persist.

The effect is large enough to quantify directly from subsample standard
deviations. Daily volatility across the calm stretch of
{window_label(CALM_WINDOW)} is **{vol_calm:.2f}%**. Across
{window_label(GFC_WINDOW)} it is **{vol_gfc:.2f}%**, and across
{window_label(COVID_WINDOW)} it is **{vol_covid:.2f}%**. Volatility in the
2020 stress window runs at roughly **{stress_ratio:.1f} times** the
calm-period level.

A single unconditional standard deviation of {daily_vol:.2f}% per day describes
neither regime. That is the precise sense in which the assumption of constant
variance is violated, and it is what motivates conditional variance models in
Notebook 05 rather than any fixed volatility estimate.
"""))
```



### Volatility clustering

The log return series exhibits volatility clustering: periods of high
volatility are followed by further high volatility, and periods of relative
calm persist.

The effect is large enough to quantify directly from subsample standard
deviations. Daily volatility across the calm stretch of
January 2013 to December 2019 is **0.81%**. Across
September 2008 to May 2009 it is **3.23%**, and across
February 2020 to May 2020 it is **3.47%**. Volatility in the
2020 stress window runs at roughly **4.3 times** the
calm-period level.

A single unconditional standard deviation of 1.21% per day describes
neither regime. That is the precise sense in which the assumption of constant
variance is violated, and it is what motivates conditional variance models in
Notebook 05 rather than any fixed volatility estimate.




```python
fig = raw['log_returns'].plot(
    title='Distribution of S&P 500 daily log returns',
    kind='hist'
)
fig.update_layout(
    showlegend=False,
    yaxis_title='Observations',
    xaxis_title='Log return'
)

# Annotation positions derive from the data so they stay on canvas and remain
# correct on a re-run. Every value shown is interpolated from a live variable.
annotation_height = raw['log_returns'].value_counts().max()
stats_box_x = raw['log_returns'].quantile(0.999)

fig.add_annotation(
    x=extreme_loss,
    y=annotation_height * 0.01,
    text=f'Largest decline: {extreme_loss_date.date()}',
    showarrow=True,
    arrowhead=1,
    ax=-40,
    ay=-40
)

fig.add_annotation(
    x=extreme_gain,
    y=annotation_height * 0.01,
    text=f'Largest gain: {extreme_gain_date.date()}',
    showarrow=True,
    arrowhead=1,
    ax=40,
    ay=-40
)

fig.add_annotation(
    x=stats_box_x,
    y=annotation_height * 0.20,
    text=(
        f'Mean = {mean_log_return:.6f}'
        f'<br>Std dev = {daily_vol:.2f}%'
        f'<br>n = {len(raw):,}'
    ),
    showarrow=False,
    bgcolor='rgba(0,0,0,0.6)',
    bordercolor='white',
    borderwidth=1,
    font=dict(size=12)
)

fig.show()
fig.write_image(
    FIGURES_DIR / 'sp500_return_distribution.png',
    width=1400,
    height=700,
    scale=2
)
```



### Distribution of log returns

The histogram is centred near zero and shows extreme values in both tails.
Those tail observations correspond to the stress periods identified above.

Unlike a normal distribution, financial returns exhibit a higher probability of
extreme events, which is a well-documented characteristic in quantitative
finance. The practical implication is that Gaussian assumptions understate tail
risk, and any risk measure built on them will understate the frequency of large
losses. This notebook establishes the pattern visually. Notebook 02 tests
normality formally through skewness, kurtosis, and the Jarque-Bera statistic.

The modelling approach that follows targets patterns in return magnitude and
volatility rather than the economic causes of individual moves.

## Stationarity test: Augmented Dickey-Fuller


```python
def adf_test(series, name='Series'):
    """Run the Augmented Dickey-Fuller test and print a decision.

    Returns the full statsmodels result tuple so the caller can reuse it.
    Returning the result avoids re-running the test to populate prose or chart
    annotations, which would otherwise mean the same statistic is computed more
    than once and quoted from separate calls.

    The p-value is rendered through fmt_pvalue rather than a format spec, so
    the printed output cannot disagree with the markdown that follows.
    """
    result = adfuller(series.dropna())

    print(f'ADF test for {name}')
    print('-' * 30)
    print(f'ADF statistic : {result[0]:.4f}')
    print(f'P-value       : {fmt_pvalue(result[1])}')
    if result[1] < 0.05:
        print('Reject null hypothesis — series is stationary')
    else:
        print('Fail to reject null hypothesis — series is not stationary')
    print(f'Lags used     : {result[2]}')
    print(f'Observations  : {result[3]:,}')
    print('Critical values:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.4f}')

    return result
```


```python
adf_result = adf_test(raw['log_returns'], 'Log returns')

adf_stat = adf_result[0]
adf_pvalue = adf_result[1]
adf_lags = adf_result[2]
adf_nobs = adf_result[3]
adf_crit = adf_result[4]
```

    ADF test for Log returns
    ------------------------------
    ADF statistic : -19.6778
    P-value       : < 0.0001
    Reject null hypothesis — series is stationary
    Lags used     : 17
    Observations  : 6,674
    Critical values:
       1%: -3.4313
       5%: -2.8620
       10%: -2.5670
    


```python
display(Markdown(f"""
### Stationarity test: Augmented Dickey-Fuller

The Augmented Dickey-Fuller test evaluates the null hypothesis that the series
contains a unit root, meaning it is non-stationary. The alternative is that the
series is stationary. The test statistic is compared against critical values
more negative than those of a standard normal distribution, because the
statistic does not follow a normal distribution under the null.

**Test results**

- ADF statistic: **{adf_stat:.4f}**
- P-value: **{fmt_pvalue(adf_pvalue)}**
- Lags included: **{adf_lags}**
- Observations used: **{adf_nobs:,}**
- Critical values: 1% **{adf_crit['1%']:.4f}**, 5% **{adf_crit['5%']:.4f}**, 10% **{adf_crit['10%']:.4f}**

The statistic of {adf_stat:.4f} is far below the 1% critical value of
{adf_crit['1%']:.4f}, so the null hypothesis of a unit root is rejected at the
1% level. Log returns are stationary in the mean.

On the p-value: statsmodels reports it through MacKinnon's approximate
interpolation, which floors at zero once the statistic falls outside the
tabulated range. The raw value returned here is {adf_pvalue}, which should be
read as below the resolution of the approximation rather than as a probability
of exactly zero. The statistic against the critical value is the defensible
form of the decision, which is why both are quoted.

This result is a necessary condition for the ARIMA models applied in Notebook
04 and for the mean specification of the GARCH-family models in Notebook 05. It
is not sufficient. Stationarity in the mean says nothing about stationarity in
variance, which the volatility clustering above already indicates is absent.
The ARCH test in Notebook 02 addresses that separately.
"""))
```



### Stationarity test: Augmented Dickey-Fuller

The Augmented Dickey-Fuller test evaluates the null hypothesis that the series
contains a unit root, meaning it is non-stationary. The alternative is that the
series is stationary. The test statistic is compared against critical values
more negative than those of a standard normal distribution, because the
statistic does not follow a normal distribution under the null.

**Test results**

- ADF statistic: **-19.6778**
- P-value: **< 0.0001**
- Lags included: **17**
- Observations used: **6,674**
- Critical values: 1% **-3.4313**, 5% **-2.8620**, 10% **-2.5670**

The statistic of -19.6778 is far below the 1% critical value of
-3.4313, so the null hypothesis of a unit root is rejected at the
1% level. Log returns are stationary in the mean.

On the p-value: statsmodels reports it through MacKinnon's approximate
interpolation, which floors at zero once the statistic falls outside the
tabulated range. The raw value returned here is 0.0, which should be
read as below the resolution of the approximation rather than as a probability
of exactly zero. The statistic against the critical value is the defensible
form of the decision, which is why both are quoted.

This result is a necessary condition for the ARIMA models applied in Notebook
04 and for the mean specification of the GARCH-family models in Notebook 05. It
is not sufficient. Stationarity in the mean says nothing about stationarity in
variance, which the volatility clustering above already indicates is absent.
The ARCH test in Notebook 02 addresses that separately.




```python
# Stationarity visual: price vs log returns

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=('S&P 500 close price', 'Daily log returns')
)

fig.add_trace(
    go.Scatter(x=raw.index, y=raw['Close'], mode='lines', name='Close price'),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=raw.index,
        y=raw['log_returns'],
        mode='lines',
        name='Log returns'
    ),
    row=2,
    col=1
)

fig.add_hline(y=0, line_dash='dash', opacity=0.6, row=2, col=1)

# Annotation interpolated from the live test result, with the p-value routed
# through the same formatter used in prose.
fig.add_annotation(
    x=0.98,
    y=0.32,
    xref='paper',
    yref='paper',
    text=(
        f'ADF statistic: {adf_stat:.4f}'
        f'<br>P-value: {fmt_pvalue(adf_pvalue)}'
        f'<br>1% critical: {adf_crit["1%"]:.4f}'
        '<br>Reject unit root'
    ),
    showarrow=False,
    align='left',
    borderwidth=1,
    bgcolor='rgba(30,30,30,0.85)',
    font=dict(size=12)
)

fig.update_layout(
    title='From price series to log returns',
    height=850,
    width=1400,
    hovermode='x unified',
    showlegend=False
)

fig.update_yaxes(title_text='Price', row=1, col=1)
fig.update_yaxes(title_text='Log return', row=2, col=1)
fig.update_xaxes(title_text='Date', row=2, col=1)

fig.show()

fig.write_image(
    FIGURES_DIR / 'stationarity_price_vs_returns.png',
    scale=3
)
```



## Rolling volatility


```python
raw[ROLLING_VOL_COL] = (
    raw['log_returns'].rolling(window=ROLLING_VOL_WINDOW).std()
)

rolling_vol_min_pct = raw[ROLLING_VOL_COL].min() * 100
rolling_vol_max_pct = raw[ROLLING_VOL_COL].max() * 100
rolling_vol_max_date = raw[ROLLING_VOL_COL].idxmax()

fig = raw[ROLLING_VOL_COL].plot(
    title=f'{ROLLING_VOL_WINDOW}-day rolling volatility (S&P 500)'
)
fig.update_layout(
    showlegend=False,
    yaxis_title='Volatility',
    xaxis_title='Date'
)
fig.show()
fig.write_image(
    FIGURES_DIR / 'sp500_rolling_volatility.png',
    width=1400,
    height=700,
    scale=2
)
```




```python
display(Markdown(f"""
### Rolling volatility ({ROLLING_VOL_WINDOW}-day)

A {ROLLING_VOL_WINDOW}-day rolling standard deviation of log returns quantifies
time-varying volatility directly. The series ranges from
**{rolling_vol_min_pct:.2f}%** to **{rolling_vol_max_pct:.2f}%** per day, the
maximum occurring on **{rolling_vol_max_date.date()}**, against an
unconditional value of **{daily_vol:.2f}%**. The peak is
**{rolling_vol_max_pct / daily_vol:.1f} times** the unconditional figure and
the trough **{rolling_vol_min_pct / daily_vol:.2f} times** it, so a single
constant estimate misstates risk in both directions depending on the period.

The rolling estimate is backward-looking and equally weighted, so it reacts to
a shock only as the shock enters the window and drops out abruptly
{ROLLING_VOL_WINDOW} sessions later. That lag is the specific limitation the
conditional variance models in Notebook 05 are designed to remove.
"""))
```



### Rolling volatility (30-day)

A 30-day rolling standard deviation of log returns quantifies
time-varying volatility directly. The series ranges from
**0.22%** to **5.41%** per day, the
maximum occurring on **2020-04-08**, against an
unconditional value of **1.21%**. The peak is
**4.5 times** the unconditional figure and
the trough **0.18 times** it, so a single
constant estimate misstates risk in both directions depending on the period.

The rolling estimate is backward-looking and equally weighted, so it reacts to
a shock only as the shock enters the window and drops out abruptly
30 sessions later. That lag is the specific limitation the
conditional variance models in Notebook 05 are designed to remove.




```python
# === DRAWDOWN ANALYSIS ===

raw['Return'] = raw['Close'].pct_change()
raw['Cumulative'] = (1 + raw['Return']).cumprod()
raw['Peak'] = raw['Cumulative'].cummax()
raw['Drawdown'] = (raw['Cumulative'] - raw['Peak']) / raw['Peak']

max_dd = raw['Drawdown'].min()
max_dd_date = raw['Drawdown'].idxmin()
current_dd = raw['Drawdown'].iloc[-1]

print(f'Maximum drawdown: {max_dd:.2%}')
print(f'Date of maximum drawdown: {max_dd_date.date()}')
print(f'Current drawdown (as of {raw.index[-1].date()}): {current_dd:.2%}')

fig = raw['Drawdown'].plot(title='S&P 500 historical drawdowns')
fig.update_layout(
    yaxis_title='Drawdown (%)',
    xaxis_title='Date',
    showlegend=False
)
fig.show()
fig.write_image(
    FIGURES_DIR / 'sp500_drawdowns.png',
    width=1400,
    height=700,
    scale=2
)

print('\nTop 5 largest drawdowns')
print(raw['Drawdown'].nsmallest(5))
```

    Maximum drawdown: -56.78%
    Date of maximum drawdown: 2009-03-09
    Current drawdown (as of 2026-08-14): -0.17%
    



    
    Top 5 largest drawdowns
    Date
    2009-03-09   -0.567754
    2009-03-05   -0.563908
    2009-03-06   -0.563377
    2009-03-03   -0.555103
    2009-03-02   -0.552235
    Name: Drawdown, dtype: float64
    


```python
display(Markdown(f"""
### Drawdown analysis

Drawdown measures the decline from a running peak, so it captures the loss an
investor would have experienced holding through a period rather than the
volatility of individual sessions. It is the risk measure most directly tied to
the experience of holding an asset.

- Maximum drawdown: **{max_dd:.2%}**, reached on **{max_dd_date.date()}**
- Current drawdown as of {raw.index[-1].date()}: **{current_dd:.2%}**

The maximum figure is a function of the sample start date. The return series
opens on {return_series_start.date()}, close to the dot-com peak, so the sample
includes a full peak-to-trough cycle from that high. A window beginning after
2003 would report a materially smaller maximum drawdown from the same
underlying index. The figure is a realised historical worst case for this
window, not a neutral forward expectation.
"""))
```



### Drawdown analysis

Drawdown measures the decline from a running peak, so it captures the loss an
investor would have experienced holding through a period rather than the
volatility of individual sessions. It is the risk measure most directly tied to
the experience of holding an asset.

- Maximum drawdown: **-56.78%**, reached on **2009-03-09**
- Current drawdown as of 2026-08-14: **-0.17%**

The maximum figure is a function of the sample start date. The return series
opens on 2000-01-04, close to the dot-com peak, so the sample
includes a full peak-to-trough cycle from that high. A window beginning after
2003 would report a materially smaller maximum drawdown from the same
underlying index. The figure is a realised historical worst case for this
window, not a neutral forward expectation.



## Calendar and seasonality analysis


```python
display(Markdown(f"""
The following analysis examines average log returns by day of week and month.
These patterns are included as descriptive context only. Seasonal components
were formally tested in Notebook 04 using SARIMA: seasonal terms were not
significant and AIC was worse than the ARIMA baseline. Seasonality was rejected
as a modelling input on that evidence, and the patterns below are not used in
any downstream model.

Dataset period: {raw.index[0].date()} to {raw.index[-1].date()}
Total sessions analysed: {len(raw):,}
"""))
```



The following analysis examines average log returns by day of week and month.
These patterns are included as descriptive context only. Seasonal components
were formally tested in Notebook 04 using SARIMA: seasonal terms were not
significant and AIC was worse than the ARIMA baseline. Seasonality was rejected
as a modelling input on that evidence, and the patterns below are not used in
any downstream model.

Dataset period: 2000-01-04 to 2026-08-14
Total sessions analysed: 6,692




```python
raw['DayOfWeek'] = raw.index.day_name()
raw['Month'] = raw.index.month_name()
raw['Year'] = raw.index.year

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

dow_returns = (
    raw.groupby('DayOfWeek')['log_returns']
    .mean()
    .reindex(day_order)
)
print('Average log return by day of week:\n', dow_returns)

month_order = [
    'January', 'February', 'March', 'April',
    'May', 'June', 'July', 'August',
    'September', 'October', 'November', 'December'
]

month_returns = (
    raw.groupby('Month')['log_returns']
    .mean()
    .reindex(month_order)
)
print('\nAverage log return by month:\n', month_returns)

fig = px.bar(
    dow_returns.reset_index(),
    x='DayOfWeek',
    y='log_returns',
    title='Average daily log return by day of week'
)
fig.show()
```

    Average log return by day of week:
     DayOfWeek
    Monday       1.488024e-04
    Tuesday      5.348646e-04
    Wednesday    3.057490e-04
    Thursday     2.489130e-04
    Friday       1.193515e-07
    Name: log_returns, dtype: float64
    
    Average log return by month:
     Month
    January     -0.000037
    February    -0.000315
    March        0.000328
    April        0.000849
    May          0.000359
    June        -0.000077
    July         0.000664
    August       0.000110
    September   -0.000714
    October      0.000534
    November     0.000984
    December     0.000247
    Name: log_returns, dtype: float64
    




```python
best_dow = dow_returns.idxmax()
best_dow_val = dow_returns.max()
worst_dow = dow_returns.idxmin()
worst_dow_val = dow_returns.min()
best_month = month_returns.idxmax()
best_month_val = month_returns.max()
worst_month = month_returns.idxmin()
worst_month_val = month_returns.min()

# The reported spread is a DIFFERENCE between two group means, so the relevant
# sampling error is the standard error of a difference, not of a single mean.
# Under roughly equal group sizes that is larger by a factor of root two.
# Scaling by the single-mean standard error would understate the spread's
# sampling error and overstate how unusual the gap looks.

dow_spread = best_dow_val - worst_dow_val
sessions_per_weekday = len(raw) / len(day_order)
se_single_mean = raw['log_returns'].std() / np.sqrt(sessions_per_weekday)
se_spread = se_single_mean * np.sqrt(2)
spread_in_se = dow_spread / se_spread

n_comparisons = len(day_order) + len(month_order)

display(Markdown(f"""
### Calendar and seasonality results

**Average log return by day of week**

- Strongest: {best_dow} ({best_dow_val:.6f})
- Weakest: {worst_dow} ({worst_dow_val:.6f})

**Average log return by month**

- Strongest: {best_month} ({best_month_val:.6f})
- Weakest: {worst_month} ({worst_month_val:.6f})

The day-of-week spread is {dow_spread:.6f}. With roughly
{sessions_per_weekday:,.0f} observations per weekday, the standard error of a
single weekday mean is about {se_single_mean:.6f}, and the standard error of
the difference between two weekday means is about {se_spread:.6f}. The observed
spread is therefore roughly **{spread_in_se:.1f} standard errors**, which is not
distinguishable from sampling noise at any conventional threshold.

The comparison is weaker still than that figure suggests. The extremes were
selected after inspecting {n_comparisons} group means, so the largest gap among
them is expected to exceed zero even under a true null of no calendar effect.
Naming the maximum and minimum of a set of noisy estimates selects on noise by
construction, and no multiple-comparison correction has been applied.

This is descriptive context. The formal test in Notebook 04 rejected
seasonality, and no downstream model uses these patterns.
"""))
```



### Calendar and seasonality results

**Average log return by day of week**

- Strongest: Tuesday (0.000535)
- Weakest: Friday (0.000000)

**Average log return by month**

- Strongest: November (0.000984)
- Weakest: September (-0.000714)

The day-of-week spread is 0.000535. With roughly
1,338 observations per weekday, the standard error of a
single weekday mean is about 0.000332, and the standard error of
the difference between two weekday means is about 0.000469. The observed
spread is therefore roughly **1.1 standard errors**, which is not
distinguishable from sampling noise at any conventional threshold.

The comparison is weaker still than that figure suggests. The extremes were
selected after inspecting 17 group means, so the largest gap among
them is expected to exceed zero even under a true null of no calendar effect.
Naming the maximum and minimum of a set of noisy estimates selects on noise by
construction, and no multiple-comparison correction has been applied.

This is descriptive context. The formal test in Notebook 04 rejected
seasonality, and no downstream model uses these patterns.



## Export


```python
# === SAVE CLEANED DATASET FOR FUTURE PHASES ===

# Core modelling dataset
core_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'log_returns']

# CSV (human-readable)
raw[core_cols].to_csv(DATA_DIR / 'sp500_cleaned.csv')

# Parquet (pipeline version, schema preserved)
raw[core_cols].to_parquet(DATA_DIR / 'sp500_cleaned.parquet')

# Full enriched dataset
raw.to_parquet(DATA_DIR / 'sp500_eda_enriched.parquet')

print('Datasets exported')
for name in (
    'sp500_cleaned.csv',
    'sp500_cleaned.parquet',
    'sp500_eda_enriched.parquet',
):
    print(f'  {DATA_DIR / name}')
```

    Datasets exported
      ..\data\sp500_cleaned.csv
      ..\data\sp500_cleaned.parquet
      ..\data\sp500_eda_enriched.parquet
    


```python
# === LOCKED METRICS EXPORT ===
#
# Single source of truth for every statistic this notebook produces. Downstream
# notebooks read these by key rather than restating values, so a figure cannot
# drift between notebooks.
#
# Structure follows the convention already established by notebooks 02 and 04
# and consumed by notebook 05: each notebook owns one nested dict under a
# notebook_NN key. Writing flat top-level keys instead would leave the file
# carrying two schemas, and a reader would have to know which notebook used
# which.
#
# Read-modify-write rather than overwrite: other notebooks write their own
# blocks to the same file, and a plain write here would discard them.

nb01_metrics = {
    # Provenance. Two start dates are recorded deliberately: the price series
    # begins one session before the return series.
    'as_of_date': str(raw.index[-1].date()),
    'price_series_start': str(price_series_start.date()),
    'return_series_start': str(return_series_start.date()),
    'span_years': float(span_years),
    'sessions_downloaded': int(sessions_before),
    'sessions_after_cleaning': int(sessions_after),
    'return_observations': int(len(raw)),
    'zero_volume_removed': int(len(zero_vol)),
    'zero_volume_date': str(zero_date.date()) if len(zero_vol) > 0 else None,
    'incomplete_session_dropped': bool(incomplete_session_dropped),
    'calendar_complete': bool(calendar_complete),
    'nyse_sessions_expected': int(len(expected_sessions)),
    'adj_close_identical_to_close': bool(adj_close_identical),

    # Statistics read downstream
    'adf_stat': float(adf_stat),
    'adf_pvalue': float(adf_pvalue),
    'adf_lags': int(adf_lags),
    'adf_nobs': int(adf_nobs),
    'adf_crit_1pct': float(adf_crit['1%']),
    'adf_crit_5pct': float(adf_crit['5%']),
    'adf_crit_10pct': float(adf_crit['10%']),
    'daily_vol_pct': float(daily_vol),
    'mean_log_return': float(mean_log_return),
    'rolling_vol_window': int(ROLLING_VOL_WINDOW),
    'rolling_vol_min_pct': float(rolling_vol_min_pct),
    'rolling_vol_max_pct': float(rolling_vol_max_pct),
    'vol_calm_pct': float(vol_calm),
    'vol_gfc_pct': float(vol_gfc),
    'vol_covid_pct': float(vol_covid),
    'stress_ratio': float(stress_ratio),
    'max_drawdown': float(max_dd),
    'max_drawdown_date': str(max_dd_date.date()),
    'current_drawdown': float(current_dd),
    'largest_daily_loss_log': float(extreme_loss),
    'largest_daily_loss_date': str(extreme_loss_date.date()),
    'largest_daily_gain_log': float(extreme_gain),
    'largest_daily_gain_date': str(extreme_gain_date.date()),
}

# Flat top-level keys written by earlier revisions of this notebook.
#
# A merge-only store never deletes, so renaming a key leaves the old one behind
# holding a value that will never update again. That is the same stale-figure
# failure this file exists to prevent, produced by the mechanism meant to
# prevent it. Any rename therefore has to be paired with an explicit purge.
LEGACY_FLAT_KEYS = {
    'adf_stat', 'adf_pvalue', 'adf_lags', 'adf_nobs',
    'adf_crit_1pct', 'adf_crit_5pct', 'adf_crit_10pct',
    'daily_vol_pct', 'mean_log_return',
    'rolling_vol_window', 'rolling_vol_min_pct', 'rolling_vol_max_pct',
    'vol_calm_pct', 'vol_gfc_pct', 'vol_covid_pct', 'stress_ratio',
    'max_drawdown', 'max_drawdown_date', 'current_drawdown',
    'largest_daily_loss_log', 'largest_daily_loss_date',
    'largest_daily_gain_log', 'largest_daily_gain_date',
}

if METRICS_PATH.exists():
    locked_metrics = json.loads(METRICS_PATH.read_text())
else:
    locked_metrics = {}

purged = sorted(
    k for k in list(locked_metrics)
    if k.startswith('nb01_') or k in LEGACY_FLAT_KEYS
)
for key in purged:
    del locked_metrics[key]

locked_metrics['notebook_01'] = nb01_metrics
METRICS_PATH.write_text(json.dumps(locked_metrics, indent=2, sort_keys=True))

other_blocks = sorted(k for k in locked_metrics if k != 'notebook_01')

print(f'Wrote {len(nb01_metrics)} keys under notebook_01 in {METRICS_PATH}')
print(f'Orphaned flat keys purged: {len(purged)}')
if purged:
    print(f'  {purged}')
print(f'Other top-level blocks preserved: {other_blocks}')
if not other_blocks:
    print('  None found. If notebooks 02 to 08 are expected to write here, '
          'confirm they use this same path.')
display(pd.Series(nb01_metrics, name='value').to_frame())
```

    Wrote 36 keys under notebook_01 in ..\data\locked_metrics.json
    Orphaned flat keys purged: 0
    Other top-level blocks preserved: ['notebook_02', 'notebook_04', 'notebook_05', 'notebook_06']
    


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
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>as_of_date</th>
      <td>2026-08-14</td>
    </tr>
    <tr>
      <th>price_series_start</th>
      <td>2000-01-03</td>
    </tr>
    <tr>
      <th>return_series_start</th>
      <td>2000-01-04</td>
    </tr>
    <tr>
      <th>span_years</th>
      <td>26.625599</td>
    </tr>
    <tr>
      <th>sessions_downloaded</th>
      <td>6695</td>
    </tr>
    <tr>
      <th>sessions_after_cleaning</th>
      <td>6694</td>
    </tr>
    <tr>
      <th>return_observations</th>
      <td>6692</td>
    </tr>
    <tr>
      <th>zero_volume_removed</th>
      <td>1</td>
    </tr>
    <tr>
      <th>zero_volume_date</th>
      <td>2023-05-24</td>
    </tr>
    <tr>
      <th>incomplete_session_dropped</th>
      <td>True</td>
    </tr>
    <tr>
      <th>calendar_complete</th>
      <td>True</td>
    </tr>
    <tr>
      <th>nyse_sessions_expected</th>
      <td>6695</td>
    </tr>
    <tr>
      <th>adj_close_identical_to_close</th>
      <td>True</td>
    </tr>
    <tr>
      <th>adf_stat</th>
      <td>-19.677837</td>
    </tr>
    <tr>
      <th>adf_pvalue</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>adf_lags</th>
      <td>17</td>
    </tr>
    <tr>
      <th>adf_nobs</th>
      <td>6674</td>
    </tr>
    <tr>
      <th>adf_crit_1pct</th>
      <td>-3.43133</td>
    </tr>
    <tr>
      <th>adf_crit_5pct</th>
      <td>-2.861973</td>
    </tr>
    <tr>
      <th>adf_crit_10pct</th>
      <td>-2.567001</td>
    </tr>
    <tr>
      <th>daily_vol_pct</th>
      <td>1.214434</td>
    </tr>
    <tr>
      <th>mean_log_return</th>
      <td>0.000251</td>
    </tr>
    <tr>
      <th>rolling_vol_window</th>
      <td>30</td>
    </tr>
    <tr>
      <th>rolling_vol_min_pct</th>
      <td>0.224483</td>
    </tr>
    <tr>
      <th>rolling_vol_max_pct</th>
      <td>5.409123</td>
    </tr>
    <tr>
      <th>vol_calm_pct</th>
      <td>0.809993</td>
    </tr>
    <tr>
      <th>vol_gfc_pct</th>
      <td>3.232693</td>
    </tr>
    <tr>
      <th>vol_covid_pct</th>
      <td>3.472553</td>
    </tr>
    <tr>
      <th>stress_ratio</th>
      <td>4.287141</td>
    </tr>
    <tr>
      <th>max_drawdown</th>
      <td>-0.567754</td>
    </tr>
    <tr>
      <th>max_drawdown_date</th>
      <td>2009-03-09</td>
    </tr>
    <tr>
      <th>current_drawdown</th>
      <td>-0.001696</td>
    </tr>
    <tr>
      <th>largest_daily_loss_log</th>
      <td>-0.127652</td>
    </tr>
    <tr>
      <th>largest_daily_loss_date</th>
      <td>2020-03-16</td>
    </tr>
    <tr>
      <th>largest_daily_gain_log</th>
      <td>0.109572</td>
    </tr>
    <tr>
      <th>largest_daily_gain_date</th>
      <td>2008-10-13</td>
    </tr>
  </tbody>
</table>
</div>



```python
display(Markdown(f"""
## EDA summary

The exploratory analysis established four properties of the S&P 500 dataset
that directly inform the modelling approach in subsequent notebooks.

**Data integrity.** {len(zero_vol)} zero-volume observation was identified and
removed, dated {zero_date.date()}. Verification against the NYSE schedule
confirmed the exchange was open that day, so a reported volume of zero is a
data integrity error rather than a market condition. The dataset index was
compared against the full NYSE trading schedule in both directions:
{'every session the exchange held is present and no dataset date falls outside the exchange calendar' if calendar_complete else 'discrepancies were found and are reported in the calendar completeness section above'}.
{'The most recent session was dropped as incomplete.' if incomplete_session_dropped else 'The most recent session was already complete and was retained.'}
The price series covers {price_series_start.date()} to {raw.index[-1].date()},
{span_years:.2f} years, from {sessions_before:,} sessions downloaded. The return
series covers {return_series_start.date()} to {raw.index[-1].date()} with
{len(raw):,} observations.

**Return transformation.** Closing prices were converted to log returns. Log
returns are additive across time, approximately symmetric for small price
changes, and better suited to statistical testing than raw price levels. The
return series begins one session after the price series, since the first
observation has no prior close against which to compute a return.

**Stationarity.** The Augmented Dickey-Fuller test rejects the null hypothesis
of a unit root: ADF statistic {adf_stat:.4f} against a 1% critical value of
{adf_crit['1%']:.4f}, p-value {fmt_pvalue(adf_pvalue)}, using {adf_lags} lags
over {adf_nobs:,} observations. Log returns are stationary in the mean, which is
a necessary condition for the time series models applied in Notebooks 04 and 05.

**Volatility structure.** Log returns have a mean of {mean_log_return:.6f},
effectively zero, and an unconditional daily standard deviation of
{daily_vol:.2f}%. Volatility is not constant: the {ROLLING_VOL_WINDOW}-day
rolling estimate ranges from {rolling_vol_min_pct:.2f}% to
{rolling_vol_max_pct:.2f}%, and subsample volatility across
{window_label(COVID_WINDOW)} runs at roughly {stress_ratio:.1f} times the
{window_label(CALM_WINDOW)} level of {vol_calm:.2f}%. Extreme moves are present
in both tails, the largest decline being {extreme_loss:.6f}
({extreme_loss_sigma:.1f} standard deviations) on {extreme_loss_date.date()}
and the largest gain {extreme_gain:.6f} ({extreme_gain_sigma:.1f} standard
deviations) on {extreme_gain_date.date()}, both far beyond what a normal
distribution fitted to this sample would admit. Maximum drawdown over the
period is {max_dd:.2%}, reached on {max_dd_date.date()}.

Autocorrelation structure, formal normality testing, and the ARCH test for
conditional heteroskedasticity are not covered in this notebook. They are the
subject of Notebook 02, and together with the volatility structure above they
motivate the GARCH-family approach in Notebook 05 over any assumption of
constant variance.

All {len(nb01_metrics)} statistics and provenance fields quoted above are
interpolated from live variables and written to `{METRICS_PATH.name}` for
downstream use.
"""))
```



## EDA summary

The exploratory analysis established four properties of the S&P 500 dataset
that directly inform the modelling approach in subsequent notebooks.

**Data integrity.** 1 zero-volume observation was identified and
removed, dated 2023-05-24. Verification against the NYSE schedule
confirmed the exchange was open that day, so a reported volume of zero is a
data integrity error rather than a market condition. The dataset index was
compared against the full NYSE trading schedule in both directions:
every session the exchange held is present and no dataset date falls outside the exchange calendar.
The most recent session was dropped as incomplete.
The price series covers 2000-01-03 to 2026-08-14,
26.63 years, from 6,695 sessions downloaded. The return
series covers 2000-01-04 to 2026-08-14 with
6,692 observations.

**Return transformation.** Closing prices were converted to log returns. Log
returns are additive across time, approximately symmetric for small price
changes, and better suited to statistical testing than raw price levels. The
return series begins one session after the price series, since the first
observation has no prior close against which to compute a return.

**Stationarity.** The Augmented Dickey-Fuller test rejects the null hypothesis
of a unit root: ADF statistic -19.6778 against a 1% critical value of
-3.4313, p-value < 0.0001, using 17 lags
over 6,674 observations. Log returns are stationary in the mean, which is
a necessary condition for the time series models applied in Notebooks 04 and 05.

**Volatility structure.** Log returns have a mean of 0.000251,
effectively zero, and an unconditional daily standard deviation of
1.21%. Volatility is not constant: the 30-day
rolling estimate ranges from 0.22% to
5.41%, and subsample volatility across
February 2020 to May 2020 runs at roughly 4.3 times the
January 2013 to December 2019 level of 0.81%. Extreme moves are present
in both tails, the largest decline being -0.127652
(-10.5 standard deviations) on 2020-03-16
and the largest gain 0.109572 (9.0 standard
deviations) on 2008-10-13, both far beyond what a normal
distribution fitted to this sample would admit. Maximum drawdown over the
period is -56.78%, reached on 2009-03-09.

Autocorrelation structure, formal normality testing, and the ARCH test for
conditional heteroskedasticity are not covered in this notebook. They are the
subject of Notebook 02, and together with the volatility structure above they
motivate the GARCH-family approach in Notebook 05 over any assumption of
constant variance.

All 36 statistics and provenance fields quoted above are
interpolated from live variables and written to `locked_metrics.json` for
downstream use.


