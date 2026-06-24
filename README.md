<p align="center">
  <img src="reports/figures/sp500_log_returns_with_vol.png" width="900"/>
</p>

# S&P 500 Market Intelligence System

**A decision-support system for market risk analysis and forecasting**

An end-to-end financial data science project that transforms historical S&P 500 market data into actionable risk intelligence.

The project combines statistical analysis, forecasting, volatility modelling, and market regime identification to answer a practical question:

> Can historical market data provide useful insight into future market risk, even when future market direction remains difficult to predict?

Built as a self-directed financial data science project, the system follows a complete analytical workflow from data validation through to forecasting, risk assessment, and decision-support outputs.

Rather than focusing solely on price prediction, the project investigates how statistical and machine learning methods can be used to identify, quantify, and communicate changing market risk conditions.

---

## 🎯 Project Objective

Financial markets are noisy, adaptive systems where short-term price direction is often difficult to predict.

However, market risk behaves differently.

Volatility exhibits persistence, clustering, and regime-dependent behaviour that can be measured, modelled, and forecasted.

The objective of this project is to build a market intelligence system capable of:

* Validating and monitoring financial data quality
* Quantifying market risk characteristics
* Forecasting future volatility
* Identifying changing market regimes
* Generating interpretable risk signals
* Supporting risk-aware investment decisions

The project follows a structured workflow:

**data validation → statistical diagnostics → feature engineering → forecasting → risk intelligence → decision support**

The emphasis is not simply producing forecasts, but translating model outputs into information that could support portfolio and risk management decisions.

---

## 🧠 Intended Decision-Support Outputs

The long-term goal is to evolve the project from a forecasting exercise into a market risk intelligence platform.

Planned outputs include:

### Market Risk Dashboard

* Current market risk assessment
* Forecast volatility reporting
* Historical risk monitoring
* Market regime tracking

### Risk Classification

Forecasts translated into interpretable risk categories:

| Risk Level | Description               |
| ---------- | ------------------------- |
| Low        | Stable market conditions  |
| Moderate   | Normal market risk        |
| High       | Elevated uncertainty      |
| Extreme    | Significant market stress |

### Market Regime Detection

Identification of market environments such as:

* Calm
* Normal
* Stress
* Crisis

### Early Warning Indicators

Signals designed to identify:

* Rising volatility
* Regime transitions
* Increasing tail risk
* Elevated market stress

### Portfolio Risk Intelligence

Future versions of the system will explore how forecast risk levels could inform:

* Portfolio exposure decisions
* Risk budgeting
* Hedging requirements
* Monitoring and review frequency

The objective is to convert statistical forecasts into actionable risk intelligence rather than treating forecasting accuracy as the final output.

---

## 📊 Key Findings (Notebooks 01–02)

### ⚠️ Data Quality & Validation

A single **zero-volume observation (2023-05-24)** was identified during a valid trading period.

Validation process:

* Cross-checked against the US market calendar
* Confirmed **not a holiday**
* Classified as a **data provider inconsistency (yfinance)**

Action taken:

* Observation removed to avoid distortions in:

  * return calculations
  * volatility estimates
  * downstream modelling

**Key insight**

> External financial data must be validated, not assumed correct.

---

### 📈 Stationarity & Time-Series Structure

Log returns were tested using the **Augmented Dickey-Fuller (ADF) test**.

Results:

* **ADF Statistic:** **-19.60**
* **P-value:** **< 0.001**

Findings:

* Log returns are **stationary**
* Raw prices are **non-stationary** and contain a unit root

This confirms that financial modelling should be performed on returns rather than raw prices.

---

### 📊 Distributional Behaviour

Statistical diagnostics reveal substantial deviation from normality.

Results:

* **Skewness:** **-0.3479**
* **Excess Kurtosis:** **10.6424**
* **Jarque–Bera Statistic:** **31,350.69**
* **P-value:** **< 0.001**

Evidence confirms:

* fat tails
* non-normality
* elevated probability of extreme outcomes
* asymmetric downside behaviour

Diagnostics included:

* Q–Q Plot analysis
* Jarque–Bera normality testing

These findings indicate that extreme market events occur more frequently than predicted under Gaussian assumptions.

**Key insight**

> Financial returns violate normality assumptions, making tail-aware risk and volatility models necessary.

---

### 📉 Volatility Structure

Volatility diagnostics reveal strong persistence and time-varying risk.

Results from Notebook 02 show:

* Weak autocorrelation in raw returns
* Strong autocorrelation in squared returns
* Significant volatility clustering
* Persistent dependence across volatility lags
* Evidence of conditional heteroskedasticity

Formal diagnostics included:

* ACF / PACF of returns
* ACF / PACF of squared returns
* Ljung–Box testing
* Engle ARCH LM testing

Key findings:

* Return direction exhibits limited predictability
* Volatility displays measurable structure and persistence
* Market risk is not randomly distributed through time

**Key insight**

> Market direction remains difficult to predict, but volatility itself displays persistent and forecastable structure.

---

### 📐 Statistical Diagnostics

Notebook 02 formally tested whether S&P 500 returns satisfy common modelling assumptions.

Findings:

* Returns are not normally distributed.
* Raw returns exhibit weak linear dependence.
* Squared returns strongly reject white-noise behaviour.
* Volatility demonstrates persistent clustering.
* ARCH effects are statistically significant.

Ljung–Box results:

* Returns show weak but statistically detectable dependence.
* Squared returns show extremely strong serial dependence.

ARCH LM results:

* **LM Statistic:** **1770.98**
* **P-value:** **< 0.001**

These diagnostics provide strong empirical support for volatility-aware modelling approaches.

**Key insight**

> Statistical significance in financial data does not always imply economic predictability, but volatility persistence creates measurable structure for risk modelling.

---

### 📉 Risk Characteristics

Historical risk analysis identified:

* **Maximum Drawdown:** **-56.78%**
* Trough date: **2009-03-09**
* Severe downside events dominate long-term risk exposure

Evidence also indicates:

* asymmetric downside risk
* heavy-tail behaviour
* elevated market stress during crisis regimes

---

### 📅 Calendar Effects

Seasonality analysis revealed weak but observable patterns.

Monthly effects:

* **Strongest:** November, April, July
* **Weakest:** September

Day-of-week effects:

* Stronger average returns on:

  * Tuesday
  * Wednesday

These effects are not strong enough for standalone trading signals but may provide useful contextual or engineered features.

---

## 📂 Notebook Roadmap

| #  | Notebook                    | Status         | Focus                                                                    |
| -- | --------------------------- | -------------- | ------------------------------------------------------------------------ |
| 01 | EDA & Data Preparation      | ✅ Completed    | Data validation, returns, volatility, drawdowns, seasonality             |
| 02 | Statistical Diagnostics     | ✅ Completed    | Distribution testing, ACF/PACF, volatility persistence, ARCH effects     |
| 03 | Feature Engineering         | ✅ Completed    | Lag features, technical indicators, volatility features, regime features |
| 04 | Classical Forecasting       | ✅ Completed    | ARIMA forecasting, benchmark comparison, forecast evaluation             |
| 05 | Volatility Modelling        | 🚧 In Progress | ARCH/GARCH models, volatility forecasting, risk estimation               |
| 06 | Deep Learning Forecasting   | ⏳ Planned      | LSTM, GRU                                                                |
| 07 | Anomaly & Regime Detection  | ⏳ Planned      | Market regimes, anomaly detection, stress identification                 |
| 08 | Decision Intelligence Layer | ⏳ Planned      | Risk scores, regime classification, portfolio risk signals               |

---

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
Forecasting & Volatility Modelling
        ↓
Regime Detection
        ↓
Risk Intelligence Layer
        ↓
Decision-Support Outputs
```

---

## 🛠 Tech Stack

### Data Processing

* Python 3.11
* pandas
* NumPy
* yfinance

### Statistical Analysis

* statsmodels
* scipy
* arch (planned)
* statistical hypothesis testing
* volatility diagnostics

### Visualisation

* Plotly (primary)
* Matplotlib

### Forecasting & Machine Learning

* ARIMA / SARIMA
* Prophet
* scikit-learn
* TensorFlow / Keras

### Environment

* Conda-based reproducible workflow
* `environment.yml`

---

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

---

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

---

## 📌 Project Status

**Active Development**

Current progress:

## 📌 Project Status

**Active Development**

Current progress:

✅ EDA & Statistical Validation
✅ Statistical Diagnostics & Stylized Facts
✅ Feature Engineering
✅ Classical Forecasting & Benchmark Evaluation
🚧 Volatility Modelling (ARCH/GARCH)
⏳ Regime Detection & Risk Intelligence


---

## Author

**Mena Beshara**

Telecommunications Engineer transitioning into Data Science and Quantitative Finance.

This project demonstrates an end-to-end financial analytics workflow covering:

* data quality validation
* statistical diagnostics
* financial time-series analysis
* volatility modelling
* forecasting and model evaluation
* risk intelligence system design

The broader objective is to build reproducible analytical systems that transform market data into interpretable insights for risk management and investment decision-making.
