<p align="center">
  <img src="reports/figures/sp500_log_returns_with_vol.png" width="900"/>
</p>

# S&P 500 Market Intelligence System

**Financial market intelligence built with data science**

An end-to-end financial data science project focused on understanding market behaviour, quantifying risk, and building a statistically sound foundation for forecasting models using S&P 500 data.

This project is not only about prediction. It is about building statistical understanding first, then modelling.

---

## 🎯 Project Objective

Financial time series are structurally different from most datasets. They are noisy, non-stationary, and influenced by regime shifts and extreme events.

This system is designed to reflect a realistic financial analytics workflow:

**data validation → statistical diagnostics → feature engineering → forecasting → evaluation**

The emphasis is on validating assumptions and understanding market behaviour before applying predictive models.

The project combines two perspectives:

- **Market behaviour and risk diagnostics**
- **Forecasting and anomaly detection**

---

## 📊 Key Findings (Phases 1–2)

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

---

### 📈 Stationarity & Time-Series Structure

Log returns were tested using the **Augmented Dickey-Fuller (ADF) test**.

Results:

- **ADF Statistic:** **-19.60**
- **P-value:** **< 0.001**

Findings:

- Log returns are **stationary**
- Raw prices are **non-stationary** and contain a unit root

This confirms that financial modelling should be performed on returns rather than raw prices.

---

### 📊 Distributional Behaviour

Return diagnostics reveal substantial deviation from normality.

Results:

- **Skewness:** **-0.3479**
- **Excess Kurtosis:** **10.6424**

Evidence confirms:

- fat tails
- non-normality
- elevated probability of extreme outcomes

These findings are reinforced through:

- Q–Q plots
- Jarque–Bera rejection of normality

---

### 📉 Volatility Structure

Volatility behaviour displays strong persistence.

Observations:

- Significant autocorrelation in **squared returns**
- Weak autocorrelation in **raw returns**
- Clear **volatility clustering**

This suggests:

> Return direction is weakly predictable, but volatility is structured and persistent.

---

### 📉 Risk Characteristics

Historical risk analysis identified:

- **Maximum Drawdown:** **-56.78%**
- Trough date: **2009-03-09**
- Severe downside events dominate long-term risk exposure

Evidence also indicates:

- asymmetric downside risk
- heavy-tail behaviour
- elevated market stress during crisis regimes

---

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

---

## 📂 Notebook Roadmap

| # | Notebook | Status | Focus |
|---|---|---|---|
| 01 | EDA & Data Preparation | ✅ Completed | Data validation, returns, volatility, drawdowns, seasonality |
| 02 | Statistical Diagnostics | 🟡 In Progress | ACF/PACF, distribution testing, volatility persistence |
| 03 | Feature Engineering | ⏳ Planned | Lag features, indicators, regime features |
| 04 | Classical Forecasting | ⏳ Planned | ARIMA, SARIMA, Prophet |
| 05 | Volatility Modelling | ⏳ Planned | GARCH family, VaR |
| 06 | Deep Learning Forecasting | ⏳ Planned | LSTM, GRU |
| 07 | Anomaly Detection | ⏳ Planned | Isolation Forest, autoencoders |
| 08 | Model Evaluation | ⏳ Planned | Backtesting and model comparison |

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

✅ EDA & Statistical Validation  
🟡 Statistical Diagnostics (in progress)  
⏳ Feature Engineering  
⏳ Forecasting & Volatility Modelling  
⏳ Deep Learning & Regime Detection

---

## 📍 Next Phase

**Feature Engineering & Forecasting Pipeline**

Upcoming work includes:

- lag structures
- technical indicators
- regime features
- GARCH volatility modelling
- ARIMA / LSTM forecasting
- model evaluation and comparison

---

## Author

**Mena Beshara**  
Telecommunications Engineer transitioning into Data Science and Quantitative Finance.

Building reproducible financial systems with a focus on:

- statistical validation
- risk analysis
- explainable modelling
- financial time-series intelligence