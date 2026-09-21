# S&P 500 Statistical Diagnostics & Stylized Facts

This notebook extends the exploratory data analysis phase by formally testing the statistical properties of S&P 500 daily log returns.

The objective is to determine whether the return series satisfies the assumptions required for time series modeling and to identify key characteristics commonly observed in financial markets.

The analysis focuses on:

- Distribution properties
- Tail behavior
- Autocorrelation structure
- Volatility persistence
- Heteroskedasticity
- White noise behavior

These diagnostics provide the statistical foundation for later forecasting, volatility modeling, and anomaly detection phases.


```python
# Import the needed libraries
import pandas as pd 
import numpy as np 
# Setting Plotly backend plotting for pandas 
pd.options.plotting.backend = 'plotly'
import plotly.io as pio
pio.templates.default = 'plotly_dark'
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path
import scipy.stats as stats
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from IPython.display import Markdown, display
import os
os.makedirs('../reports/figures', exist_ok=True)

```

## Import Required Libraries

The following libraries are used for:

- Statistical testing
- Distribution analysis
- Autocorrelation diagnostics
- Volatility diagnostics
- Visualization

These tools allow us to formally evaluate the statistical behavior of financial return series.

## Load Data


```python
df = pd.read_parquet('../data/sp500_cleaned.parquet').copy()
# preview Dataset
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



## Load Cleaned Dataset

The dataset used in this notebook was cleaned and validated during the EDA phase.

This includes:

- Removal of anomalous observations
- Validation of trading dates
- Log return calculation
- Consistent datetime indexing

Using a pre-cleaned dataset ensures reproducibility and prevents duplication of preprocessing steps across notebooks.


```python
# Dataset information
df.info()
```

    <class 'pandas.DataFrame'>
    DatetimeIndex: 6692 entries, 2000-01-04 to 2026-08-14
    Data columns (total 6 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Open         6692 non-null   float64
     1   High         6692 non-null   float64
     2   Low          6692 non-null   float64
     3   Close        6692 non-null   float64
     4   Volume       6692 non-null   int64  
     5   log_returns  6692 non-null   float64
    dtypes: float64(5), int64(1)
    memory usage: 366.0 KB
    


```python
# Check Index Type
type(df.index)
```




    pandas.DatetimeIndex




```python
# Summary Statistics
df.describe()
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
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6692.000000</td>
      <td>6692.000000</td>
      <td>6692.000000</td>
      <td>6692.000000</td>
      <td>6.692000e+03</td>
      <td>6692.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2381.511862</td>
      <td>2394.992908</td>
      <td>2367.057454</td>
      <td>2381.918148</td>
      <td>3.468390e+09</td>
      <td>0.000251</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1614.573690</td>
      <td>1622.055625</td>
      <td>1606.695060</td>
      <td>1615.016584</td>
      <td>1.533962e+09</td>
      <td>0.012144</td>
    </tr>
    <tr>
      <th>min</th>
      <td>679.280029</td>
      <td>695.270020</td>
      <td>666.789978</td>
      <td>676.530029</td>
      <td>3.560700e+08</td>
      <td>-0.127652</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1215.365021</td>
      <td>1222.982544</td>
      <td>1208.927460</td>
      <td>1215.645020</td>
      <td>2.366590e+09</td>
      <td>-0.004712</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1582.554993</td>
      <td>1590.854980</td>
      <td>1577.325012</td>
      <td>1583.929993</td>
      <td>3.561515e+09</td>
      <td>0.000639</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2996.427429</td>
      <td>3006.067505</td>
      <td>2978.112488</td>
      <td>2995.714966</td>
      <td>4.325190e+09</td>
      <td>0.005885</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7806.600098</td>
      <td>7816.700195</td>
      <td>7776.310059</td>
      <td>7798.990234</td>
      <td>1.145623e+10</td>
      <td>0.109572</td>
    </tr>
  </tbody>
</table>
</div>



## Dataset Validation

Before performing statistical diagnostics, the dataset structure is briefly validated to confirm:

- Correct datetime indexing
- Presence of return series
- Numerical consistency
- Expected observation count

This step ensures the dataset is properly prepared for statistical testing.

# Distribution Diagnostics 

## 1) Skewness & Kurtosis


```python
skewness = df['log_returns'].skew()
kurtosis = df['log_returns'].kurtosis()

print(f'Skewness: {skewness:.4f}')
print(f'Kurtosis: {kurtosis:.4f}')
```

    Skewness: -0.3493
    Kurtosis: 10.6566
    


```python
display(Markdown(
f"""
## 📊 Interpretation of Skewness & Kurtosis Results

### Results

- **Skewness:** **{skewness:.4f}**
- **Excess Kurtosis:** **{kurtosis:.4f}**

### Interpretation of Skewness

The negative skewness indicates that the return distribution is slightly tilted toward larger downside moves.

This means:

- Negative returns tend to be more extreme than positive returns.
- Market declines are often sharper and more sudden than rallies.
- Downside tail risk is more pronounced.

This behaviour is common in equity markets, where panic selling and systemic shocks can produce rapid drawdowns.

### Interpretation of Kurtosis

The excess kurtosis value of **{kurtosis:.4f}** is substantially above zero.

The high kurtosis observed here indicates:

- The distribution contains **fat tails**.
- Extreme returns occur far more frequently than predicted by Gaussian assumptions.
- Market crashes and large rallies are statistically more common than standard models would expect.

Financial return series often violate normal distribution assumptions due to:

- Fat tails.
- Extreme market events.
- Time-varying volatility.

These characteristics motivate the use of more advanced volatility and risk models in later stages of the project.

### Key Insight

> Extreme market movements occur more frequently than predicted by standard normal distribution assumptions.
"""
))
```



## 📊 Interpretation of Skewness & Kurtosis Results

### Results

- **Skewness:** **-0.3493**
- **Excess Kurtosis:** **10.6566**

### Interpretation of Skewness

The negative skewness indicates that the return distribution is slightly tilted toward larger downside moves.

This means:

- Negative returns tend to be more extreme than positive returns.
- Market declines are often sharper and more sudden than rallies.
- Downside tail risk is more pronounced.

This behaviour is common in equity markets, where panic selling and systemic shocks can produce rapid drawdowns.

### Interpretation of Kurtosis

The excess kurtosis value of **10.6566** is substantially above zero.

The high kurtosis observed here indicates:

- The distribution contains **fat tails**.
- Extreme returns occur far more frequently than predicted by Gaussian assumptions.
- Market crashes and large rallies are statistically more common than standard models would expect.

Financial return series often violate normal distribution assumptions due to:

- Fat tails.
- Extreme market events.
- Time-varying volatility.

These characteristics motivate the use of more advanced volatility and risk models in later stages of the project.

### Key Insight

> Extreme market movements occur more frequently than predicted by standard normal distribution assumptions.



# Q-Q Plot (Quantile-Quantile Plot)


```python

# Generate Q-Q plot data
(osm, osr), (slope, intercept, r) = stats.probplot(
    df['log_returns'],
    dist='norm'
)

# Create Plotly figure
fig = go.Figure()

# Sample quantiles
fig.add_trace(
    go.Scatter(
        x=osm,
        y=osr,
        mode='markers',
        name='Observed Returns',
        marker=dict(size=4)
    )
)

# Reference normal line
fig.add_trace(
    go.Scatter(
        x=osm,
        y=slope * np.array(osm) + intercept,
        mode='lines',
        name='Normal Distribution Reference'
    )
)

# Layout
fig.update_layout(
    title='Q-Q Plot: S&P 500 Log Returns vs Normal Distribution',
    xaxis_title='Theoretical Quantiles',
    yaxis_title='Observed Quantiles',
    template='plotly_dark',
    height=700,
    width=900,
    legend_title=''
)

# Show plot
fig.show()

# Save figure
fig.write_image(
    "../reports/figures/sp500_qq_plot.png",
    width=1200,
    height=900,
    scale=2
)
```



# The Jarque-Bera Test


```python
# Unpack the test results for clarity
jb_stat, p_value = stats.jarque_bera(df['log_returns'])

# Print results
print(f"Jarque-Bera Statistic: {jb_stat:,.2f}")
print(f"P-Value:               {p_value:.4f}")

# Add a simple logic check for the Null Hypothesis
alpha = 0.05
if p_value < alpha:
    print("\nResult: Reject H0 - The distribution is NOT normal (Fat Tails confirmed).")
else:
    print("\nResult: Fail to reject H0 - The distribution appears normal.")
```

    Jarque-Bera Statistic: 31,748.52
    P-Value:               0.0000
    
    Result: Reject H0 - The distribution is NOT normal (Fat Tails confirmed).
    


```python
display(Markdown(
f"""
### 📊 Jarque-Bera Normality Test

The Jarque-Bera test formally assesses whether the data has skewness and kurtosis consistent with a normal distribution.

**Results:**

- **Jarque-Bera Statistic:** **{jb_stat:,.2f}**
- **P-Value:** **{p_value:.3g}**

**Conclusion:** We {'fail to reject' if p_value >= 0.05 else 'strongly reject'} the null hypothesis of normality.

### Key Insight

Because S&P 500 returns exhibit **fat tails** and **volatility clustering**:

- Traditional VaR and Expected Shortfall (ES) models assuming normality can **significantly underestimate tail risk**.
- This may lead to insufficient capital buffers during periods of market stress.
- **GARCH-family models** address this limitation by explicitly modelling time-varying volatility and volatility clustering, providing more realistic risk forecasts.

These diagnostics provide a statistical justification for moving from simple return models to more sophisticated volatility models in Notebook 05.
"""
))
```



### 📊 Jarque-Bera Normality Test

The Jarque-Bera test formally assesses whether the data has skewness and kurtosis consistent with a normal distribution.

**Results:**

- **Jarque-Bera Statistic:** **31,748.52**
- **P-Value:** **0**

**Conclusion:** We strongly reject the null hypothesis of normality.

### Key Insight

Because S&P 500 returns exhibit **fat tails** and **volatility clustering**:

- Traditional VaR and Expected Shortfall (ES) models assuming normality can **significantly underestimate tail risk**.
- This may lead to insufficient capital buffers during periods of market stress.
- **GARCH-family models** address this limitation by explicitly modelling time-varying volatility and volatility clustering, providing more realistic risk forecasts.

These diagnostics provide a statistical justification for moving from simple return models to more sophisticated volatility models in Notebook 05.



# Serial Dependence Diagnostics 

## ACF of Returns


```python
# Compute ACF + confidence intervals
acf_values = acf(
    df['log_returns'],
    nlags=40
)
n = len(df['log_returns'].dropna())
conf = 1.96 / np.sqrt(n)

lags = np.arange(len(acf_values))

fig = go.Figure()

# Bars
fig.add_trace(
    go.Bar(
        x=lags,
        y=acf_values,
        name='ACF'
    )
)
fig.add_hline(y=conf, line_dash='dash', line_color='gray')
fig.add_hline(y=-conf, line_dash='dash', line_color='gray')

fig.update_layout(
    title='ACF of S&P 500 Log Returns',
    xaxis_title='Lag',
    yaxis_title='Autocorrelation',
    template='plotly_dark'
)
fig.write_image(
    "../reports/figures/sp500_acf_returns.png",
    width=1400,
    height=700,
    scale=2
)
fig.show()
```



### 📊 Autocorrelation Analysis (ACF) — Log Returns

The Autocorrelation Function (ACF) examines whether current log returns are correlated with their past values at different lags.

**Observations**
- Most lags show autocorrelation values very close to zero.
- A small but statistically significant **negative autocorrelation** appears at **lag 1**.
- No slow decay or strong persistent patterns are visible across higher lags.

**Interpretation**

The S&P 500 daily log returns exhibit **very weak serial correlation**. 

While the slight negative lag-1 autocorrelation is statistically significant, its economic magnitude is small. This aligns with the **weak-form Efficient Market Hypothesis** — past price movements provide limited predictive power for future directional moves.

**Important Nuance**

> Absence of strong autocorrelation in **returns** does **not** mean the market is purely random.

Financial time series frequently show much stronger dependence in:
- **Squared returns** (volatility clustering)
- **Absolute returns**
- Conditional variance

This is why we often observe near-zero ACF in raw returns but significant autocorrelation in squared returns — a key stylized fact that motivates **GARCH-type models**.

**Key Insight**

> Daily returns are largely unpredictable from past returns alone, but **volatility** (risk) itself tends to persist over time.

# PACF of Returns


```python
# Compute PACF
pacf_vals = pacf(
    df['log_returns'].dropna(),
    nlags = 40,
    method = 'yw'
) 
# Confidence interval
n = len(df['log_returns'].dropna())
conf = 1.96 / np.sqrt(n)

# Plot
fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=list(range(len(pacf_vals))),
        y=pacf_vals,
        name='PACF'
    )
)

# Confidence bands
fig.add_hline(y=conf, line_dash='dash')
fig.add_hline(y=-conf, line_dash='dash')
fig.add_hline(y=0)

fig.update_layout(
    title='PACF of S&P 500 Log Returns',
    xaxis_title='Lag',
    yaxis_title='Partial Autocorrelation',
    showlegend=False
)
fig.write_image(
    "../reports/figures/sp500_pacf_returns.png",
    width=1400,
    height=700,
    scale=2
)

fig.show()
```



### 📊 Partial Autocorrelation Function (PACF) — Log Returns

The Partial Autocorrelation Function (PACF) measures the **direct correlation** between the current log return and its lagged values, after removing the effects of all intermediate lags.

**Observations**
- The most notable feature is a **statistically significant negative partial autocorrelation at lag 1**.
- Partial autocorrelations at higher lags are generally small and mostly fall within the 95% confidence bands.
- There is no clear pattern of slowly decaying or repeating significant lags beyond lag 1.

**Interpretation**

The negative PACF at lag 1 indicates a mild **mean-reverting tendency** in daily returns: 
- A positive return today is slightly more likely to be followed by a negative return tomorrow (and vice versa), after accounting for other lags.

This short-term negative dependence is a known characteristic in equity index returns and can be attributed to:
- Market overreaction and subsequent correction
- Bid-ask bounce effects
- Portfolio rebalancing and profit-taking behavior

However, the economic magnitude of this effect is small, consistent with the overall weak serial dependence observed in the ACF.

**Key Insight**

> While S&P 500 daily returns exhibit very limited directional predictability overall, there is a detectable **short-term negative autocorrelation at lag 1**, suggesting mild mean reversion at the daily horizon. 

This finding reinforces the earlier conclusion from the ACF: past returns contain only weak information for forecasting future returns — a result aligned with weak-form market efficiency.

# ACF / PACF of Squared Returns


```python
# Create squared returns 
df['squared_returns'] = df['log_returns'] ** 2

# Check results 
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
      <th>squared_returns</th>
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
      <td>1.528746e-03</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>1399.420044</td>
      <td>1413.270020</td>
      <td>1377.680054</td>
      <td>1402.109985</td>
      <td>1085500000</td>
      <td>0.001920</td>
      <td>3.687698e-06</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>1402.109985</td>
      <td>1411.900024</td>
      <td>1392.099976</td>
      <td>1403.449951</td>
      <td>1092300000</td>
      <td>0.000955</td>
      <td>9.124486e-07</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>1403.449951</td>
      <td>1441.469971</td>
      <td>1400.729980</td>
      <td>1441.469971</td>
      <td>1225200000</td>
      <td>0.026730</td>
      <td>7.144902e-04</td>
    </tr>
    <tr>
      <th>2000-01-10</th>
      <td>1441.469971</td>
      <td>1464.359985</td>
      <td>1441.469971</td>
      <td>1457.599976</td>
      <td>1064800000</td>
      <td>0.011128</td>
      <td>1.238285e-04</td>
    </tr>
  </tbody>
</table>
</div>



### 📊 Squared Returns

To study volatility behavior, log returns are squared.

Unlike raw returns, squared returns remove market direction and focus on the **magnitude of price movements**.

This allows us to examine whether periods of:

- Large market movements  
- Elevated risk  
- Relative market calm  

tend to persist over time.

If squared returns exhibit autocorrelation, this suggests **volatility clustering** — a defining characteristic of financial markets.

**Why this matters**

Traditional return autocorrelation focuses on **directional predictability**.

Squared returns instead focus on **risk persistence**.

A market may show little predictability in price direction while still displaying persistent and forecastable volatility.


```python
# Compute acf for squared returns 
lags = 40   # 2 Months 
acf_vals = acf(df['squared_returns'].dropna(), nlags= lags)

# Confidence interval
n = len(df['log_returns'].dropna())
conf = 1.96 / np.sqrt(n)

# Plot
fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=list(range(len(acf_vals))),
        y=acf_vals,
        name='ACF'
    )
)

# Confidence bands
fig.add_hline(y=conf, line_dash='dash')
fig.add_hline(y=-conf, line_dash='dash')
fig.add_hline(y=0)

fig.update_layout(
    title='ACF of Squared S&P 500 Log Returns',
    xaxis_title='Lag',
    yaxis_title='Autocorrelation',
    showlegend=False
)

fig.show()
fig.write_image(
    "../reports/figures/acf_squared_returns.png",
    width=1400,
    height=700,
    scale=2
)
```



### 📊 ACF of Squared Returns — Volatility Persistence

The Autocorrelation Function (ACF) of squared returns examines whether **volatility is correlated with its own past values**.

Unlike the ACF of raw returns, which measures directional dependence, squared-return autocorrelation focuses on the **persistence of market risk**.

**Lag Selection**

The number of lags determines how far back volatility dependence is examined.

For daily financial data, using **20–40 lags is common practice**:

- **20 lags** ≈ roughly **1 trading month**  
- **40 lags** ≈ roughly **2 trading months**  

This range provides a practical balance:

- Long enough to detect **volatility persistence and clustering**  
- Short enough to avoid excessive statistical noise and difficult interpretation  

In this analysis, **40 lags** were used to evaluate whether volatility dependence persists beyond short-term market fluctuations.

**Observations**

- A large number of lags exceed the confidence intervals  
- Autocorrelation remains statistically significant across many lags  
- The autocorrelation gradually decays over time rather than disappearing immediately  

**Interpretation**

This pattern provides strong evidence of:

- **Volatility clustering**  
- **Time-varying variance (heteroskedasticity)**  
- Persistent dependence in return magnitude  

The gradual decay is particularly important.

If volatility were independent over time, autocorrelation would fluctuate randomly around zero.

Instead, the slow decline suggests that volatility exhibits **memory**, where periods of elevated market risk tend to persist before eventually reverting toward calmer conditions.

This behavior is one of the most widely documented **stylized facts of financial markets**.

**Key Insight**

> While market direction may appear difficult to predict, **volatility itself displays persistent and measurable structure**.

# PACF of Squared Returns


```python
# Compute PACF
pacf_vals = pacf(
    df['squared_returns'].dropna(),
    nlags = 40,
    method = 'yw'
) 
# Confidence interval
n = len(df['log_returns'].dropna())
conf = 1.96 / np.sqrt(n)

# Plot
fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=list(range(len(pacf_vals))),
        y=pacf_vals,
        name='PACF'
    )
)

# Confidence bands
fig.add_hline(y=conf, line_dash='dash')
fig.add_hline(y=-conf, line_dash='dash')
fig.add_hline(y=0)

fig.update_layout(
    title='PACF of Squared S&P 500 Log Returns',
    xaxis_title='Lag',
    yaxis_title='Partial Autocorrelation',
    showlegend=False
)

fig.show()
fig.write_image(
    "../reports/figures/pacf_squared_returns.png",
    width=1400,
    height=700,
    scale=2
)
```



### 📊 PACF of Squared Returns — Direct Volatility Dependence

The Partial Autocorrelation Function (PACF) measures the **direct relationship** between squared returns and prior lags after controlling for intermediate lag effects.

While ACF captures total dependence across lags, PACF isolates the **direct contribution** of individual lags.

**Observations**

- Statistically significant autocorrelation appears primarily in the **first 4–5 lags**  
- After these early lags, partial autocorrelation declines rapidly and moves within the confidence intervals  
- No long sequence of significant direct lag effects is observed  

**Interpretation**

This result suggests that volatility dependence is driven mainly by **recent market activity**.

The first few lagged volatility observations contain most of the explanatory information, while more distant lags contribute relatively little once earlier volatility effects are accounted for.

This explains an important difference between the ACF and PACF results:

- **ACF** shows persistent volatility dependence across many lags  
- **PACF** indicates that the strongest **direct** effects occur primarily at short horizons  

This combination is commonly observed in financial time series and is consistent with **ARCH/GARCH-type volatility processes**.

**Why this matters**

The PACF results suggest that volatility is not influenced equally by all historical periods.

Instead, recent volatility carries the strongest direct influence on future volatility behavior.

This provides additional statistical justification for using **volatility models that explicitly account for short-term persistence and changing variance**.

**Key Insight**

> Financial volatility often displays **short-term direct dependence combined with longer-term persistence**, a characteristic that motivates ARCH and GARCH volatility modeling.

# Ljung-Box Test on Returns


```python
# Ljung-Box test on log returns
# I test:
# first 10 trading days
# first 20 trading days
lb_returns = acorr_ljungbox(
    df['log_returns'],
    lags=[10, 20],
    return_df=True
)

lb_returns
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
      <td>97.329282</td>
      <td>1.863275e-16</td>
    </tr>
    <tr>
      <th>20</th>
      <td>162.143366</td>
      <td>2.893678e-24</td>
    </tr>
  </tbody>
</table>
</div>




```python
display(Markdown(
f"""
### 📊 Ljung-Box Test — Serial Dependence in Log Returns

The Ljung-Box test formally evaluates whether the return series behaves as **white noise**.

Unlike the ACF plot, which examines individual lags separately, the Ljung-Box test evaluates whether autocorrelation exists **jointly across multiple lags**.

### Hypotheses

**Null Hypothesis (H₀)**

The return series is white noise, meaning no autocorrelation exists up to the tested lag.

**Alternative Hypothesis (H₁)**

At least one lag exhibits statistically significant autocorrelation.

### Results

| Lags | Ljung-Box Statistic | P-value |
|---|---:|---:|
| 10 | **{lb_returns.loc[10, 'lb_stat']:.2f}** | **{lb_returns.loc[10, 'lb_pvalue']:.3g}** |
| 20 | **{lb_returns.loc[20, 'lb_stat']:.2f}** | **{lb_returns.loc[20, 'lb_pvalue']:.3g}** |

### Interpretation

- Both p-values are {'substantially below' if (lb_returns['lb_pvalue'] < 0.05).all() else 'not consistently below'} **0.05**.
- The null hypothesis of white noise is {'rejected' if (lb_returns['lb_pvalue'] < 0.05).all() else 'not rejected'}.
- This indicates {'the presence of statistically significant serial dependence within the return series.' if (lb_returns['lb_pvalue'] < 0.05).all() else 'insufficient evidence of statistically significant serial dependence within the return series.'}

### Important Nuance

This result does **not** contradict the earlier ACF analysis.

The ACF of returns showed mostly weak individual correlations, with only small spikes exceeding the confidence intervals. The Ljung-Box test instead evaluates whether these small correlations, when considered **collectively**, are statistically distinguishable from pure randomness.

In large financial datasets, even economically small autocorrelations may become statistically significant.

### Key Insight

> S&P 500 returns are not perfectly white noise statistically, although their linear predictability remains economically limited.

This reinforces an important finance principle:

- **Statistical significance ≠ economic significance**

Weak dependence may exist in returns, but it does not necessarily translate into exploitable forecasting power.
"""
))
```



### 📊 Ljung-Box Test — Serial Dependence in Log Returns

The Ljung-Box test formally evaluates whether the return series behaves as **white noise**.

Unlike the ACF plot, which examines individual lags separately, the Ljung-Box test evaluates whether autocorrelation exists **jointly across multiple lags**.

### Hypotheses

**Null Hypothesis (H₀)**

The return series is white noise, meaning no autocorrelation exists up to the tested lag.

**Alternative Hypothesis (H₁)**

At least one lag exhibits statistically significant autocorrelation.

### Results

| Lags | Ljung-Box Statistic | P-value |
|---|---:|---:|
| 10 | **97.33** | **1.86e-16** |
| 20 | **162.14** | **2.89e-24** |

### Interpretation

- Both p-values are substantially below **0.05**.
- The null hypothesis of white noise is rejected.
- This indicates the presence of statistically significant serial dependence within the return series.

### Important Nuance

This result does **not** contradict the earlier ACF analysis.

The ACF of returns showed mostly weak individual correlations, with only small spikes exceeding the confidence intervals. The Ljung-Box test instead evaluates whether these small correlations, when considered **collectively**, are statistically distinguishable from pure randomness.

In large financial datasets, even economically small autocorrelations may become statistically significant.

### Key Insight

> S&P 500 returns are not perfectly white noise statistically, although their linear predictability remains economically limited.

This reinforces an important finance principle:

- **Statistical significance ≠ economic significance**

Weak dependence may exist in returns, but it does not necessarily translate into exploitable forecasting power.



# Ljung-Box on squared returns


```python
# Ljung-Box test on squared returns
# I test:
# first 10 trading days
# first 20 trading days
ljung_box_squared = acorr_ljungbox(
    df['squared_returns'],
    lags=[10, 20],
    return_df=True
)

ljung_box_squared
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
      <td>6107.520069</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>8607.602101</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
display(Markdown(
f"""
### 📊 Ljung-Box Test — Volatility Dependence in Squared Returns

The Ljung-Box test was applied to **squared log returns** to determine whether volatility behaves as white noise or exhibits persistent dependence over time.

Unlike raw returns, squared returns capture the **magnitude of market movements**, making them a useful proxy for volatility dynamics.

### Hypotheses

**Null Hypothesis (H₀)**

Squared returns are white noise.

This implies:

- No volatility clustering.
- Constant variance over time.
- No ARCH-type effects.

**Alternative Hypothesis (H₁)**

Squared returns exhibit autocorrelation.

This implies:

- Volatility persistence.
- Time-varying variance.
- Potential ARCH/GARCH structure.

### Results

| Lags | Ljung-Box Statistic | P-value |
|---|---:|---:|
| 10 | **{ljung_box_squared.loc[10, 'lb_stat']:.2f}** | **{ljung_box_squared.loc[10, 'lb_pvalue']:.3g}** |
| 20 | **{ljung_box_squared.loc[20, 'lb_stat']:.2f}** | **{ljung_box_squared.loc[20, 'lb_pvalue']:.3g}** |

### Interpretation

- Both p-values are {'effectively zero' if (ljung_box_squared['lb_pvalue'] < 1e-10).all() else 'well below 0.05' if (ljung_box_squared['lb_pvalue'] < 0.05).all() else 'not consistently below 0.05'}.
- The null hypothesis of white noise is {'decisively rejected' if (ljung_box_squared['lb_pvalue'] < 0.05).all() else 'not rejected'}.
- Squared returns exhibit {'extremely strong serial dependence.' if (ljung_box_squared['lb_pvalue'] < 0.05).all() else 'no statistically significant serial dependence.'}

This indicates that volatility is **not randomly distributed through time**.

Instead, periods of elevated volatility tend to be followed by further elevated volatility, while calm periods also persist.

These findings formally confirm the volatility clustering already observed in the ACF and PACF diagnostics.

### Comparison with Raw Returns

The Ljung-Box test on raw returns detected only modest serial dependence.

In contrast, the squared-return test produced substantially larger test statistics:

- Returns: **{lb_returns.loc[10, 'lb_stat']:.0f}–{lb_returns.loc[20, 'lb_stat']:.0f}**
- Squared returns: **{ljung_box_squared.loc[10, 'lb_stat']:.0f}–{ljung_box_squared.loc[20, 'lb_stat']:.0f}**

This difference highlights a central stylised fact of financial markets:

> Market direction may be difficult to predict, but market volatility often exhibits strong persistence.

### Key Insight

The S&P 500 return series exhibits **conditional heteroskedasticity**, meaning variance changes over time rather than remaining constant.

This provides strong statistical justification for:

- **ARCH testing**
- **GARCH-family volatility models**
- Forecasting conditional risk rather than price direction alone.
"""
))
```



### 📊 Ljung-Box Test — Volatility Dependence in Squared Returns

The Ljung-Box test was applied to **squared log returns** to determine whether volatility behaves as white noise or exhibits persistent dependence over time.

Unlike raw returns, squared returns capture the **magnitude of market movements**, making them a useful proxy for volatility dynamics.

### Hypotheses

**Null Hypothesis (H₀)**

Squared returns are white noise.

This implies:

- No volatility clustering.
- Constant variance over time.
- No ARCH-type effects.

**Alternative Hypothesis (H₁)**

Squared returns exhibit autocorrelation.

This implies:

- Volatility persistence.
- Time-varying variance.
- Potential ARCH/GARCH structure.

### Results

| Lags | Ljung-Box Statistic | P-value |
|---|---:|---:|
| 10 | **6107.52** | **0** |
| 20 | **8607.60** | **0** |

### Interpretation

- Both p-values are effectively zero.
- The null hypothesis of white noise is decisively rejected.
- Squared returns exhibit extremely strong serial dependence.

This indicates that volatility is **not randomly distributed through time**.

Instead, periods of elevated volatility tend to be followed by further elevated volatility, while calm periods also persist.

These findings formally confirm the volatility clustering already observed in the ACF and PACF diagnostics.

### Comparison with Raw Returns

The Ljung-Box test on raw returns detected only modest serial dependence.

In contrast, the squared-return test produced substantially larger test statistics:

- Returns: **97–162**
- Squared returns: **6108–8608**

This difference highlights a central stylised fact of financial markets:

> Market direction may be difficult to predict, but market volatility often exhibits strong persistence.

### Key Insight

The S&P 500 return series exhibits **conditional heteroskedasticity**, meaning variance changes over time rather than remaining constant.

This provides strong statistical justification for:

- **ARCH testing**
- **GARCH-family volatility models**
- Forecasting conditional risk rather than price direction alone.



# ARCH Effect Test (Engle ARCH LM Test)


```python
lm_stat, lm_pval, f_stat, f_pval = het_arch(df['log_returns'], nlags=10)

print(f"LM Statistic : {lm_stat:.4f}")
print(f"LM P-Value   : {lm_pval:.6f}")
print(f"F Statistic  : {f_stat:.4f}")
print(f"F P-Value    : {f_pval:.6f}")

if lm_pval < 0.05:
    print("\nResult: Reject H0 — ARCH effects present. Volatility is time-varying.")
else:
    print("\nResult: Fail to reject H0 — No significant ARCH effects detected.")
```

    LM Statistic : 1786.7044
    LM P-Value   : 0.000000
    F Statistic  : 243.4808
    F P-Value    : 0.000000
    
    Result: Reject H0 — ARCH effects present. Volatility is time-varying.
    


```python
display(Markdown(
f"""
### 📊 ARCH Effect Test (Engle's ARCH LM Test)

The ARCH test evaluates whether the variance of returns is constant over time or whether volatility depends on past volatility.

This is formally a test for **conditional heteroskedasticity**.

### Hypotheses

**Null Hypothesis (H₀):**

No ARCH effects exist.

→ Volatility is constant over time (homoskedasticity).

**Alternative Hypothesis (H₁):**

ARCH effects are present.

→ Volatility changes over time and depends on past shocks.

### Test Results

- **LM Statistic:** **{lm_stat:.2f}**
- **LM P-Value:** **{lm_pval:.3g}**
- **F Statistic:** **{f_stat:.2f}**
- **F P-Value:** **{f_pval:.3g}**

### Interpretation

The p-values are {'effectively zero' if lm_pval < 1e-10 else 'well below 0.05' if lm_pval < 0.05 else 'greater than 0.05'}, allowing us to **{'strongly reject' if lm_pval < 0.05 else 'fail to reject'}** the null hypothesis of constant variance.

{'This provides strong statistical evidence that S&P 500 volatility is **time-varying** and exhibits **conditional heteroskedasticity**.' if lm_pval < 0.05 else 'This provides insufficient statistical evidence to conclude that volatility is time-varying due to ARCH effects.'}

This result is consistent with earlier diagnostics:

- Volatility clustering observed in returns.
- Strong autocorrelation in squared returns.
- Significant Ljung-Box statistics on squared returns.
- Fat-tailed return distribution.

### Why This Matters

Financial returns often show limited predictability in **direction**, but substantially greater predictability in **volatility**.

This means:

- Returns may resemble near-white-noise processes.
- Risk and volatility are not random.
- Periods of high volatility tend to persist.

This behaviour violates the assumptions of constant-variance models and directly motivates the use of **ARCH/GARCH-family volatility models**.

### Key Insight

> The market may be difficult to predict, but volatility itself displays persistent structure that can be modelled and forecast.
"""
))
```



### 📊 ARCH Effect Test (Engle's ARCH LM Test)

The ARCH test evaluates whether the variance of returns is constant over time or whether volatility depends on past volatility.

This is formally a test for **conditional heteroskedasticity**.

### Hypotheses

**Null Hypothesis (H₀):**

No ARCH effects exist.

→ Volatility is constant over time (homoskedasticity).

**Alternative Hypothesis (H₁):**

ARCH effects are present.

→ Volatility changes over time and depends on past shocks.

### Test Results

- **LM Statistic:** **1786.70**
- **LM P-Value:** **0**
- **F Statistic:** **243.48**
- **F P-Value:** **0**

### Interpretation

The p-values are effectively zero, allowing us to **strongly reject** the null hypothesis of constant variance.

This provides strong statistical evidence that S&P 500 volatility is **time-varying** and exhibits **conditional heteroskedasticity**.

This result is consistent with earlier diagnostics:

- Volatility clustering observed in returns.
- Strong autocorrelation in squared returns.
- Significant Ljung-Box statistics on squared returns.
- Fat-tailed return distribution.

### Why This Matters

Financial returns often show limited predictability in **direction**, but substantially greater predictability in **volatility**.

This means:

- Returns may resemble near-white-noise processes.
- Risk and volatility are not random.
- Periods of high volatility tend to persist.

This behaviour violates the assumptions of constant-variance models and directly motivates the use of **ARCH/GARCH-family volatility models**.

### Key Insight

> The market may be difficult to predict, but volatility itself displays persistent structure that can be modelled and forecast.




```python
display(Markdown(
f"""
## ✅ Statistical Diagnostics Summary

This notebook formally evaluated the statistical properties of S&P 500 daily log returns to assess modelling assumptions and document the key stylised facts commonly observed in financial markets.

### Key Findings

**1. Return Distribution**

The log return distribution is clearly non-normal. It exhibits a skewness of **{skewness:.4f}** and an excess kurtosis of **{kurtosis:.4f}**, confirming the presence of **fat tails**. Extreme market events occur more frequently than would be expected under Gaussian assumptions.

**2. Serial Dependence in Returns**

ACF, PACF, and Ljung-Box diagnostics indicate very weak autocorrelation in raw returns. Although the Ljung-Box test {'rejects' if (lb_returns['lb_pvalue'] < 0.05).all() else 'does not reject'} the white-noise hypothesis, the observed autocorrelation remains economically small and does not indicate strong or persistent linear dependence.

This behaviour is broadly consistent with **weak-form market efficiency**, where past returns provide limited predictive power for future directional movements.

**3. Volatility Clustering**

Squared-return diagnostics reveal a substantially different structure.

- The ACF of squared returns shows strong autocorrelation that gradually decays over time.
- The PACF indicates volatility dependence concentrated in the early lags.
- The Ljung-Box test on squared returns {'strongly rejects' if (ljung_box_squared['lb_pvalue'] < 0.05).all() else 'does not reject'} the white-noise hypothesis.

These findings confirm **volatility clustering**, where periods of elevated market activity tend to persist.

**4. Conditional Heteroskedasticity**

Engle's ARCH LM test {'strongly rejected' if lm_pval < 0.05 else 'did not reject'} the null hypothesis of constant variance (**LM Statistic = {lm_stat:.2f}, p-value = {lm_pval:.3g}**), providing {'strong evidence' if lm_pval < 0.05 else 'insufficient evidence'} of **conditional heteroskedasticity**.

Volatility in the S&P 500 is therefore **{'time-varying rather than constant' if lm_pval < 0.05 else 'not shown to vary significantly over time based on this test'}**.

### Final Interpretation

S&P 500 daily returns exhibit limited predictability in direction but clear statistical structure in volatility and risk.

While linear dependence in returns is weak, volatility demonstrates persistence, clustering, and significant memory.

These results provide strong empirical justification for moving beyond constant-variance assumptions and towards **volatility-aware modelling approaches**, particularly **ARCH/GARCH-family models**, which are explored in the next phase of the project.

### Key Takeaway

> Market direction remains difficult to forecast, but volatility itself exhibits persistent structure that can be modelled and forecast.
"""
))
```



## ✅ Statistical Diagnostics Summary

This notebook formally evaluated the statistical properties of S&P 500 daily log returns to assess modelling assumptions and document the key stylised facts commonly observed in financial markets.

### Key Findings

**1. Return Distribution**

The log return distribution is clearly non-normal. It exhibits a skewness of **-0.3493** and an excess kurtosis of **10.6566**, confirming the presence of **fat tails**. Extreme market events occur more frequently than would be expected under Gaussian assumptions.

**2. Serial Dependence in Returns**

ACF, PACF, and Ljung-Box diagnostics indicate very weak autocorrelation in raw returns. Although the Ljung-Box test rejects the white-noise hypothesis, the observed autocorrelation remains economically small and does not indicate strong or persistent linear dependence.

This behaviour is broadly consistent with **weak-form market efficiency**, where past returns provide limited predictive power for future directional movements.

**3. Volatility Clustering**

Squared-return diagnostics reveal a substantially different structure.

- The ACF of squared returns shows strong autocorrelation that gradually decays over time.
- The PACF indicates volatility dependence concentrated in the early lags.
- The Ljung-Box test on squared returns strongly rejects the white-noise hypothesis.

These findings confirm **volatility clustering**, where periods of elevated market activity tend to persist.

**4. Conditional Heteroskedasticity**

Engle's ARCH LM test strongly rejected the null hypothesis of constant variance (**LM Statistic = 1786.70, p-value = 0**), providing strong evidence of **conditional heteroskedasticity**.

Volatility in the S&P 500 is therefore **time-varying rather than constant**.

### Final Interpretation

S&P 500 daily returns exhibit limited predictability in direction but clear statistical structure in volatility and risk.

While linear dependence in returns is weak, volatility demonstrates persistence, clustering, and significant memory.

These results provide strong empirical justification for moving beyond constant-variance assumptions and towards **volatility-aware modelling approaches**, particularly **ARCH/GARCH-family models**, which are explored in the next phase of the project.

### Key Takeaway

> Market direction remains difficult to forecast, but volatility itself exhibits persistent structure that can be modelled and forecast.




```python
# Save this notebook Matrics for future calls 
metrics_path = Path('../data/locked_metrics.json')
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
 
metrics['notebook_02'] = {
    'skewness': float(skewness),
    'excess_kurtosis': float(kurtosis),
    'jarque_bera_stat': float(jb_stat),
    'jarque_bera_p': float(p_value),
    'arch_lm_stat': float(lm_stat),
    'arch_lm_p': float(lm_pval),
    'arch_lm_nlags': 10,
}
 
metrics_path.write_text(json.dumps(metrics, indent=2))
print(f"Exported notebook_02 metrics to {metrics_path.resolve()}")
for k, v in metrics['notebook_02'].items():
    print(f"  {k}: {v}")
```

    Exported notebook_02 metrics to C:\Users\Mena\Documents\Python\sp500-market-intelligence\data\locked_metrics.json
      skewness: -0.3492648905612519
      excess_kurtosis: 10.656592911706545
      jarque_bera_stat: 31748.522549048226
      jarque_bera_p: 0.0
      arch_lm_stat: 1786.704404754513
      arch_lm_p: 0.0
      arch_lm_nlags: 10
    
