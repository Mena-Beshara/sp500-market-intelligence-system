<p align="center">
  <img src="reports/figures/sp500_log_returns_with_vol.png" width="900"/>
</p>

# S&P 500 Market Intelligence System

**Financial market intelligence built with data science**

An end-to-end financial data science project focused on understanding market behaviour, quantifying risk, and building a statistically grounded foundation for forecasting models using S&P 500 data.

Built as a self-directed financial data science system to demonstrate end-to-end analytical, statistical, and forecasting capability.

This project prioritises statistical understanding and market structure before predictive modelling.

## 🎯 Project Objective

Financial time series are structurally different from most datasets. They are noisy, non-stationary, and influenced by regime shifts and extreme events.

This system is designed to reflect a realistic financial analytics workflow:

**data validation → statistical diagnostics → feature engineering → forecasting → evaluation**

The emphasis is on validating assumptions and understanding market behaviour before applying predictive models.

The project combines two perspectives:

- **Market behaviour and risk diagnostics**
- **Forecasting and anomaly detection**

## 📊 Key Findings (Notebooks 01–02)

### ⚠️ Data Quality & Validation

A single **zero-volume observation (2023-05-24)** was identified during a valid trading period.

Validation process:

- Cross-checked against the US market calendar
- Confirmed **not a holiday**
- Classified as a **data provider inconsistency (yfinance)**

Action taken:

- Observation removed to avoid distortions in:
  - return calculations
  - volatility estimates
  - downstream modelling

**Key insight**

> External financial data must be validated, not assumed correct.

### 📈 Stationarity & Time-Series Structure

Log returns were tested using the **Augmented Dickey-Fuller (ADF) test**.

Results:

- **ADF Statistic:** **-19.60**
- **P-value:** **< 0.001**

Findings:

- Log returns are **stationary**
- Raw prices are **non-stationary** and contain a unit root

This confirms that financial modelling should be performed on returns rather than raw prices.

### 📊 Distributional Behaviour

Statistical diagnostics reveal substantial deviation from normality.

Results:

- **Skewness:** **-0.3479**
- **Excess Kurtosis:** **10.6424**
- **Jarque–Bera Statistic:** **31,350.69**
- **P-value:** **< 0.001**

Evidence confirms:

- fat tails
- non-normality
- elevated probability of extreme outcomes
- asymmetric downside behaviour

Diagnostics included:

- Q–Q Plot analysis
- Jarque–Bera normality testing

These findings indicate that extreme market events occur more frequently than predicted under Gaussian assumptions.

**Key insight**

> Financial returns violate normality assumptions, making tail-aware risk and volatility models necessary.

### 📉 Volatility Structure

Volatility diagnostics reveal strong persistence and time-varying risk.

Results from Notebook 02 show:

- Weak autocorrelation in raw returns
- Strong autocorrelation in squared returns
- Significant volatility clustering
- Persistent dependence across volatility lags
- Evidence of conditional heteroskedasticity

Formal diagnostics included:

- ACF / PACF of returns
- ACF / PACF of squared returns
- Ljung–Box testing
- Engle ARCH LM testing

Key findings:

- Return direction exhibits limited predictability
- Volatility displays measurable structure and persistence
- Market risk is not randomly distributed through time

**Key insight**

> Market direction remains difficult to predict, but volatility itself displays persistent and forecastable structure.

### 📐 Statistical Diagnostics

Notebook 02 formally tested whether S&P 500 returns satisfy common modelling assumptions.

Findings:

- Returns are not normally distributed.
- Raw returns exhibit weak linear dependence.
- Squared returns strongly reject white-noise behaviour.
- Volatility demonstrates persistent clustering.
- ARCH effects are statistically significant.

Ljung–Box results:

- Returns show weak but statistically detectable dependence.
- Squared returns show extremely strong serial dependence.

ARCH LM results:

- **LM Statistic:** **1770.98**
- **P-value:** **< 0.001**

These diagnostics provide strong empirical support for volatility-aware modelling approaches.

**Key insight**

> Statistical significance in financial data does not always imply economic predictability, but volatility persistence creates measurable structure for risk modelling.

### 📉 Risk Characteristics

Historical risk analysis identified:

- **Maximum Drawdown:** **-56.78%**
- Trough date: **2009-03-09**
- Severe downside events dominate long-term risk exposure

Evidence also indicates:

- asymmetric downside risk
- heavy-tail behaviour
- elevated market stress during crisis regimes

### 📅 Calendar Effects

Seasonality analysis revealed weak but observable patterns.

Monthly effects:

- **Strongest:** November, April, July
- **Weakest:** September

Day-of-week effects:

- Stronger average returns on:
  - Tuesday
  - Wednesday

These effects are not strong enough for standalone trading signals but may provide useful contextual or engineered features.

## 📂 Notebook Roadmap

| # | Notebook | Status | Focus |
|---|---|---|---|
| 01 | EDA & Data Preparation | ✅ Completed | Data validation, returns, volatility, drawdowns, seasonality |
| 02 | Statistical Diagnostics | ✅ Completed | Distribution testing, ACF/PACF, volatility persistence, ARCH effects |
| 03 | Feature Engineering | ⏳ Planned | Lag features, indicators, regime features |
| 04 | Classical Forecasting | ⏳ Planned | ARIMA, SARIMA, Prophet |
| 05 | Volatility Modelling | ⏳ Planned | GARCH family, VaR |
| 06 | Deep Learning Forecasting | ⏳ Planned | LSTM, GRU |
| 07 | Anomaly Detection | ⏳ Planned | Isolation Forest, autoencoders |
| 08 | Model Evaluation | ⏳ Planned | Backtesting and model comparison |

## 🔄 Project Workflow

```text
Raw Market Data
        ↓
Data Validation & Cleaning
        ↓
EDA & Statistical Diagnostics
        ↓
Feature Engineering
        ↓
Classical Forecasting
        ↓
Volatility Modelling
        ↓
Deep Learning
        ↓
Anomaly & Regime Detection
        ↓
Model Evaluation
```

## 🛠 Tech Stack

### Data Processing
- Python 3.11
- pandas
- NumPy
- yfinance

### Statistical Analysis
- statsmodels
- scipy
- arch (planned)
- statistical hypothesis testing
- volatility diagnostics

### Visualisation
- Plotly (primary)
- Matplotlib

### Forecasting & Machine Learning (Planned)
- ARIMA / SARIMA
- Prophet
- scikit-learn
- TensorFlow / Keras

### Environment
- Conda-based reproducible workflow
- `environment.yml`

## 📁 Repository Structure

```text
sp500-market-intelligence/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_statistical_diagnostics.ipynb
│   └── ...
│
├── data/
│   ├── sp500_cleaned.csv
│   ├── sp500_cleaned.parquet
│   └── sp500_eda_enriched.parquet
│
├── reports/
│   └── figures/
│
├── environment.yml
└── README.md
```

## 🚀 How to Run

```bash
# Clone repository
git clone https://github.com/Mena-Beshara/sp500-market-intelligence.git

cd sp500-market-intelligence

# Create environment
conda env create -f environment.yml
conda activate sp500-intel

# Launch notebooks
jupyter lab
```

## 📌 Project Status

**Active Development**

Current progress:

✅ EDA & Statistical Validation  
✅ Statistical Diagnostics & Stylized Facts  
⏳ Feature Engineering  
⏳ Forecasting & Volatility Modelling  
⏳ Deep Learning & Regime Detection

## 📍 Next Phase

**Feature Engineering & Market Signal Construction**

Upcoming work includes:

- lag structures
- rolling statistical features
- technical indicators
- calendar and regime features
- volatility-aware signal engineering
- preparation for forecasting and GARCH modelling

## Author

**Mena Beshara**  
Telecommunications Engineer transitioning into Data Science and Quantitative Finance.

Building reproducible financial intelligence systems with a focus on:

Building reproducible financial intelligence systems with a focus on:

- statistical validation
- statistical diagnostics
- risk analysis
- explainable modelling
- financial time-series intelligence