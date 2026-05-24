<p align="center">
  <img src="reports/figures/sp500_log_returns_with_vol.png" width="900"/>
</p>

# S&P 500 Market Intelligence System

**A financial data science and market intelligence project** combining **forecasting, volatility modelling, and anomaly detection** — built to demonstrate production-grade workflows for fintech, banking, and quantitative risk roles.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.x-orange)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-4CAF50)
![yfinance](https://img.shields.io/badge/Data-yfinance-yellow)

---

## Project Overview

This project analyses the **S&P 500 Index (^GSPC)** from 2000–present using a structured financial data science pipeline.

The system combines two complementary objectives:

### 1. Forecasting
Building and comparing:
- Classical time-series models
- Volatility models
- Deep learning approaches

### 2. Market Risk & Regime Intelligence
Understanding:
- volatility behaviour
- market stress regimes
- anomaly detection
- downside risk
- statistical market structure

The goal is to demonstrate:

- data diligence  
- statistical rigor  
- financial domain understanding  
- reproducible workflows  
- model validation and interpretability

---

## Current Progress

### Phase 1 — Exploratory Data Analysis & Statistical Validation ✅

Completed work includes:

- Data integrity validation
- OHLCV structure interpretation
- Multi-index cleaning & datetime pipeline
- Zero-volume anomaly investigation
- Daily log returns construction
- Stationarity validation (ADF)
- Rolling volatility analysis
- Volatility clustering identification
- Drawdown analysis
- Seasonality exploration
- Statistical diagnostics & stylized facts

Next phase:

➡️ Feature Engineering & Forecasting Pipeline

---

## Project Workflow

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

---

## Notebook Structure

| # | Notebook | Description |
|---|---|---|
| **01** | [`01_eda.ipynb`](01_eda.ipynb) | **EDA, Data Integrity & Financial Foundations**<br>Data loading, OHLCV interpretation, datetime pipeline, zero-volume validation, log returns, stationarity, volatility, drawdowns, seasonality |
| **02** | [`02_statistical_diagnostics.ipynb`](02_statistical_diagnostics.ipynb) | **Statistical Diagnostics & Stylized Facts**<br>Distribution analysis, skewness/kurtosis, Q-Q plots, Jarque-Bera, ACF/PACF, Ljung-Box, ARCH effects, white-noise testing |
| **03** | [`03_feature_engineering.ipynb`](03_feature_engineering.ipynb) | **Feature Engineering & Technical Indicators**<br>Rolling statistics, lag features, RSI, MACD, ATR, Bollinger Bands, calendar effects, regime indicators |
| **04** | [`04_classical_forecasting.ipynb`](04_classical_forecasting.ipynb) | **Classical Forecasting**<br>ARIMA/SARIMA family, Prophet, baseline models, walk-forward validation |
| **05** | [`05_garch_volatility_modeling.ipynb`](05_garch_volatility_modeling.ipynb) | **Volatility Modelling**<br>ARCH/GARCH, EGARCH, GJR-GARCH, conditional volatility, VaR |
| **06** | [`06_deep_learning.ipynb`](06_deep_learning.ipynb) | **Deep Learning Forecasting**<br>LSTM, GRU, sequence modelling, multi-step prediction |
| **07** | [`07_anomaly_detection.ipynb`](07_anomaly_detection.ipynb) | **Anomaly & Regime Detection**<br>Isolation Forest, Autoencoders, extreme-event detection, market regimes |
| **08** | [`08_model_evaluation.ipynb`](08_model_evaluation.ipynb) | **Evaluation & Comparison**<br>RMSE, MAE, directional accuracy, residual diagnostics, backtesting |

---

## Repository Structure

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

## Tech Stack

### Core
- Python 3.11
- pandas
- NumPy
- yfinance

### Visualisation
- Plotly
- Matplotlib / Seaborn

### Statistical & Forecasting
- statsmodels
- pmdarima
- Prophet
- arch

### Machine Learning
- scikit-learn
- TensorFlow / Keras

### Environment
- Conda (`environment.yml`)
- reproducible notebook workflow

---

## How to Run

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

## Project Status

**Active Development**  
Currently progressing through:

✅ EDA & Statistical Validation  
🔄 Feature Engineering  
⏳ Forecasting & Volatility Modelling  
⏳ Deep Learning & Regime Detection

---

## Author

**Mena Beshara**  
Telecommunications Engineer transitioning into Data Science & Quantitative Finance.

Building reproducible financial systems with a focus on statistical validation, risk analysis, and explainable modelling.
