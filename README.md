# S&P 500 Market Intelligence System

A quantitative decision-support system designed to support systematic long-term investing.

The project transforms historical and live S&P 500 market data into a structured daily Market Risk Report that helps evaluate portfolio risk before new capital is allocated. Rather than predicting market direction or generating trading signals, the system estimates market conditions through statistical analysis, volatility forecasting and market regime classification.

The objective is to replace intuition with a repeatable, evidence-based investment process.

---

## The problem

Financial markets are noisy, non-linear and difficult to predict.

Classical forecasting techniques often perform no better than simple benchmarks when applied to daily returns, yet market volatility exhibits persistent statistical structure through clustering and changing regimes.

This project asks a different question.

> **Can historical market data be transformed into interpretable risk information that supports a repeatable investment decision process?**

Rather than attempting to predict direction, the project focuses on estimating uncertainty.

---

## The system

```text
Historical Market Data
          │
          ▼
Data Validation
          │
          ▼
Statistical Diagnostics
          │
          ▼
Feature Engineering
          │
          ▼
Volatility Forecasting
          │
          ▼
Market Regime Classification
          │
          ▼
Risk Intelligence
          │
          ▼
Daily Market Risk Report
          │
          ▼
Decision Log
```

Every stage contributes to a single objective: producing a transparent, evidence-based assessment of current market risk.

---

## Daily Market Risk Report

```text
-------------------------------------------------
S&P 500 Market Risk Report
-------------------------------------------------

Date:                  2026-07-01

Forecast volatility:   23.8% annualised
Historical percentile: 89th
Market regime:         Stress
95% Value at Risk:     -2.1%
Anomaly detected:      No

Risk summary

Market volatility remains elevated relative to
recent history. Although current conditions fall
within the historical distribution, continued
portfolio monitoring is recommended before the
next trading session.

Decision logged.

-------------------------------------------------
```

The report deliberately separates model outputs from investment decisions.

The model estimates market conditions.

The investor remains responsible for every portfolio decision.

---

## Current capabilities

### Data engineering

* Download historical S&P 500 market data
* Validate data quality before analysis
* Detect missing observations and duplicates
* Verify trading calendar consistency
* Export validated datasets for downstream modelling

### Statistical analysis

* Stationarity testing
* Distribution analysis
* Fat-tail analysis
* Volatility clustering diagnostics
* Drawdown analysis
* Correlation structure
* Risk metric calculation

### Feature engineering

* Log returns
* Rolling volatility
* Lagged features
* Momentum features
* Calendar variables
* Bias-free feature construction

### Forecasting

* Historical Mean benchmark
* ARIMA evaluation
* SARIMA evaluation
* Prophet evaluation
* Walk-forward validation

### Volatility modelling (in progress)

* Realised volatility forecasting
* GARCH models
* Persistence benchmark
* HAR-RV benchmark
* Rolling parameter stability
* Residual diagnostics

### Risk intelligence (planned)

* Market regime classification
* Risk score generation
* Daily risk report
* Decision logging
* Portfolio monitoring metrics

---

## Research findings

The direction forecasting stage produced one result that shaped the rest of the system.

Forecasting market direction using ARIMA, SARIMA and Prophet did not outperform a simple Historical Mean benchmark under walk-forward evaluation.

Rather than introducing additional model complexity without evidential justification, the project shifted its modelling effort toward volatility forecasting, where the statistical diagnostics demonstrated persistent structure through conditional heteroskedasticity.

The project now estimates market risk rather than predicting returns.

---

## Research workflow

| Stage                    | Objective                        | Status |
| ------------------------ | -------------------------------- | :----: |
| Data validation          | Validate raw market data         |   ✅   |
| Statistical diagnostics  | Characterise return behaviour    |   ✅   |
| Feature engineering      | Create forecasting features      |   ✅   |
| Direction forecasting    | Evaluate predictive edge         |   ✅   |
| Volatility modelling     | Forecast conditional volatility  |   🚧   |
| Deep learning comparison | Compare against GARCH            |   📋   |
| Anomaly detection        | Detect structural market changes |   📋   |
| Risk intelligence        | Generate Market Risk Report      |   📋   |

---

## Methodology

* Data quality is verified before modelling.
* Look-ahead bias is explicitly prevented.
* Walk-forward validation is preferred over random train-test splits.
* Simpler models are preferred unless additional complexity improves out-of-sample performance.
* Every modelling decision is supported by statistical evidence.
* Every stage is reproducible and suitable for technical review.

---

## Technology stack

### Languages

* Python 3.11

### Data

* pandas
* NumPy
* Parquet
* yfinance

### Statistical modelling

* statsmodels
* SciPy
* arch

### Visualisation

* Plotly

### Development

* Jupyter Notebook
* Conda
* environment.yml

---

## Repository structure

```text
market-intelligence-system/

│
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_statistical_diagnostics.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_direction_forecasting.ipynb
│   ├── 05_volatility_modelling.ipynb
│   ├── 06_deep_learning_comparison.ipynb
│   ├── 07_anomaly_detection.ipynb
│   └── 08_risk_intelligence.ipynb
│
├── data/
│
├── environment.yml
│
└── README.md
```

The repository currently focuses on the research phase. Once the analytical work is complete, the notebooks will be modularised into a production-ready Python package.

---

## Production roadmap

### Research phase

* ✅ Data validation
* ✅ Statistical diagnostics
* ✅ Feature engineering
* ✅ Direction forecasting
* 🚧 Volatility forecasting
* 📋 Deep learning comparison
* 📋 Anomaly detection
* 📋 Risk intelligence system

### Production phase

* Modular `src` package
* Daily report generation
* Unit testing
* Model Card
* Automated execution
* Configuration management

---

## Known limitations

The system estimates market risk rather than market direction.

Forecasts are derived from historical observations and cannot anticipate unforeseen macroeconomic or geopolitical events.

Market regime thresholds are percentile-based and may require recalibration as market structure evolves.

The current implementation is designed around SPY as a representative long-term equity position and should not be assumed to generalise to other asset classes without additional validation.

The project produces decision-support information only. It does not generate trading signals or investment advice.

---

## Why this project

Many market forecasting projects stop once a prediction has been generated.

This project continues one step further.

It investigates whether statistical models can support a disciplined investment process rather than simply producing forecasts. Every model output feeds into a structured Market Risk Report, every investment decision is logged separately from the model output, and the framework can be evaluated retrospectively to determine whether the decision process improves consistency over time.

The emphasis is not on predicting the future. It is on building a transparent, repeatable and auditable decision-support system for long-term investing.

It also demonstrates a complete quantitative workflow:

* validating financial data,
* performing statistical diagnostics,
* engineering forecasting features,
* evaluating competing models,
* selecting models based on evidence,
* translating model outputs into business decisions.

---

## License

This repository is intended for educational and portfolio purposes.

Nothing in this repository constitutes financial or investment advice. The Market Intelligence System provides quantitative decision-support information only, and all investment decisions remain the responsibility of the investor.