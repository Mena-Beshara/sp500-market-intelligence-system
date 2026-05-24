# S&P 500 Market Intelligence System

A reproducible **financial market intelligence and forecasting system** built using the **S&P 500 Index (^GSPC)**.

This project combines **market diagnostics, risk analysis, and forecasting workflows** to study how financial markets behave and how statistically validated models can be built on top of that understanding.

Designed to demonstrate practical capabilities relevant to:

- Data Science
- Quantitative / Risk Analytics
- Forecasting & Time Series Modeling
- FinTech & Banking Analytics
- Machine Learning Engineering (forecasting systems)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.x-orange)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-4CAF50)
![yfinance](https://img.shields.io/badge/Data-yfinance-yellow)

---

# Executive Summary

Most financial projects jump directly into prediction.

This project takes a different approach.

Before forecasting, I validate whether the market data itself is statistically suitable for modeling.

The system combines:

### Market Intelligence
Understanding:

- market structure
- volatility behavior
- drawdowns
- seasonality
- statistical properties
- extreme events

### Forecasting
Building toward:

- ARIMA / SARIMA
- volatility models (GARCH family)
- deep learning forecasting
- model comparison and validation

The objective is not simply to fit models.

It is to build a **reproducible, statistically grounded forecasting pipeline** supported by financial reasoning and transparent methodology.

---

# Why This Project Exists

Financial forecasting is highly sensitive to:

- data quality
- structural assumptions
- volatility behavior
- distributional properties

Ignoring these foundations often produces:

- unstable models
- misleading relationships
- weak out-of-sample performance

This project approaches forecasting the same way many quantitative and risk workflows operate in practice:

**validate first, model second.**

---

# Current Progress

## Phase 1 — Exploratory Data Analysis & Statistical Validation ✅

Completed work includes:

### Data Validation & Cleaning
- DatetimeIndex construction
- Multi-level column handling from `yfinance`
- OHLCV interpretation
- Zero-volume anomaly investigation
- Data integrity validation

### Statistical Foundations
- Daily log returns
- Distribution analysis
- Stationarity validation (ADF)
- Unit-root diagnostics
- Rolling volatility
- Volatility clustering
- Drawdown analysis
- Seasonality exploration

### Statistical Diagnostics *(in progress)*
- ACF / PACF
- Jarque–Bera normality testing
- Tail-risk exploration
- Distribution diagnostics

### Outputs
- Cleaned dataset export (`Parquet`)
- Reproducible figures
- Structured notebook workflow

---

# Key Findings So Far

## Statistical Validation

### Stationarity

Raw prices are unsuitable for direct modeling due to non-stationary behavior.

After transforming prices into **daily log returns**, formal testing provided strong evidence against a unit root.

**ADF Statistic:** **-19.60**  
**P-Value:** **< 0.001**

This supports a statistically suitable foundation for classical time-series modeling.

---

## Volatility & Market Structure

Rolling volatility analysis confirms **volatility clustering**.

Periods of market stress show persistent increases in volatility, while calmer regimes exhibit lower sustained risk.

This matters because volatility persistence violates constant-variance assumptions and motivates volatility-aware modeling approaches.

---

## Tail Risk & Extreme Events

Large market moves significantly exceed typical daily fluctuations.

Notable periods include:

### Largest Losses
- March 16, 2020 (~-12.7%)
- March 12, 2020
- 2008 Financial Crisis

### Largest Gains
- 2008 relief rallies
- March 2020 COVID rebounds
- April 2025 macro-driven volatility

These observations highlight:

- fat tails
- asymmetric risk
- extreme-event behavior

all of which are important characteristics of financial return series.

---

## Drawdown Risk

Historical drawdown analysis identified:

**Maximum Drawdown:** **-56.78%**

This provides a practical measure of downside risk and helps contextualize market stress before forecasting.

---

## Seasonality

Historical return patterns show measurable calendar effects.

### Average Log Return by Day

**Strongest**
- Tuesday (**0.000528**)
- Wednesday (**0.000326**)

**Weakest**
- Friday (**0.000011**)

### Average Log Return by Month

**Strongest**
- November (**0.000984**)
- April (**0.000849**)
- July (**0.000693**)

**Weakest**
- September (**-0.000714**)
- February (**-0.000315**)

These patterns do not imply guaranteed trading opportunities but help reveal long-run market tendencies.

---

## Data Integrity Matters

During validation, a **zero-volume observation** was identified on:

**May 24, 2023**

Verification confirmed:

- not a market holiday
- valid trading day

Because the S&P 500 is an **index rather than a directly traded asset**, volume is derived and dependent on the data provider.

The zero-volume value was therefore treated as a **data integrity issue** rather than a market condition and removed from the dataset.

### Key Insight

> External financial data should be validated, not blindly trusted.

---

# Project Workflow

```text
Market Data Acquisition
            ↓
Data Validation & Cleaning
            ↓
EDA & Statistical Validation
            ↓
Statistical Diagnostics
            ↓
Feature Engineering
            ↓
Forecasting & Volatility Modeling
            ↓
Deep Learning
            ↓
Anomaly / Regime Detection
            ↓
Evaluation & Comparison
```

---

# Notebook Roadmap

| Notebook | Focus | Topics |
|---|---|---|
| **01_eda.ipynb** | EDA & Market Foundations | Data validation, OHLCV, log returns, stationarity, volatility, drawdowns, seasonality |
| **02_statistical_diagnostics.ipynb** | Diagnostics | ACF/PACF, Jarque–Bera, Q-Q plots, skewness, kurtosis, ARCH, Ljung–Box |
| **03_feature_engineering.ipynb** | Feature Engineering | Lag features, rolling statistics, technical indicators |
| **04_classical_forecasting.ipynb** | Classical Forecasting | ARIMA/SARIMA, walk-forward validation |
| **05_garch_volatility_modeling.ipynb** | Volatility Models | ARCH/GARCH, conditional volatility |
| **06_deep_learning.ipynb** | Deep Learning | LSTM / GRU |
| **07_anomaly_detection.ipynb** | Market Intelligence | Regime detection, anomalies |
| **08_model_evaluation.ipynb** | Evaluation | RMSE, MAE, backtesting |

---

# Repository Structure

```text
sp500-market-intelligence/
│
├── notebooks/
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   └── figures/
├── environment.yml
└── README.md
```

---

# Tech Stack

## Core
- Python 3.11
- pandas
- NumPy
- yfinance

## Visualisation
- Plotly
- Matplotlib
- Seaborn

## Statistical & Forecasting
- statsmodels
- pmdarima
- Prophet
- arch

## Machine Learning
- scikit-learn
- TensorFlow / Keras

## Reproducibility
- Conda (`environment.yml`)
- Version-controlled notebooks
- Exported figures and datasets

---

# Running the Project

```bash
git clone https://github.com/Mena-Beshara/sp500-market-intelligence.git

cd sp500-market-intelligence

conda env create -f environment.yml

conda activate sp500-intel

jupyter lab
```

---

# Roadmap

Planned future work includes:

- Feature Engineering
- ARIMA / SARIMA forecasting
- Volatility modeling (GARCH)
- Deep learning forecasting
- Market anomaly & regime detection
- Comparative model evaluation

Potential future additions:

- Streamlit dashboard
- Forecasting API

---

# About Me

**Mena Beshara**

Telecommunications Engineer transitioning into **Data Science and Quantitative Finance**.

Building reproducible forecasting and market intelligence systems grounded in:

- statistical validation
- financial reasoning
- risk analysis
- transparent methodology
- explainable modeling

GitHub:
https://github.com/Mena-Beshara