# ML-Driven Supply Chain Control Tower

![Supply Chain Control Tower](images/banner.png)

> An end-to-end supply-chain analytics system that turns retail and supplier data into **demand forecasts, inventory policies, supplier-risk signals, financial exposure, and prioritized actions**.

**Python · XGBoost · SHAP · PostgreSQL · Streamlit · Supply Chain Analytics**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](#)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-EC6A00)](#)
[![PostgreSQL](https://img.shields.io/badge/SQL-PostgreSQL-4169E1?logo=postgresql&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](#)

## Overview

Supply-chain teams need more than an isolated forecasting model. They need a workflow that connects **demand → inventory → suppliers → financial impact → action**.

This project implements that workflow on a synthetic retail dataset. A reproducible Python pipeline generates analytics and ML outputs, while a Streamlit dashboard turns those outputs into operational decision support.

### What the system answers

- **What is happening?** — sales, inventory, supplier, and financial KPIs
- **What is likely to happen?** — 30-day demand forecasts at store × product level
- **What needs attention?** — stockout, low-coverage, replenishment, and supplier-risk signals
- **What should we do?** — forecast-driven replenishment recommendations
- **What could it cost?** — lost-sales exposure, holding cost, and scenario-based impact

---

## Architecture

```text
                 Synthetic Retail + Supplier Data
                               │
              ┌────────────────┴────────────────┐
              │                                 │
        PostgreSQL / SQL                 Python Analytics
              │                                 │
              └────────────────┬────────────────┘
                               │
                     Demand Forecasting
                  Baselines + XGBoost + SHAP
                               │
                    Chronological Validation
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
   Inventory Policy      Supplier Risk         Financials
   Safety Stock / ROP    KPI + Temporal       Revenue / Margin
   EOQ / Replenishment  Signals              / Exposure
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                    30-Day Policy Simulation
                               │
                       Decision Support
                               │
                     Streamlit Control Tower
```

The design deliberately keeps the project focused on one decision workflow rather than presenting unrelated ML algorithms.

---

## Results

Current pipeline results from the included synthetic dataset:

| Metric | Result |
|---|---:|
| Daily records | **109,650** |
| Store × product pairs | **150** |
| XGBoost MAE | **5.7937** |
| XGBoost RMSE | **8.2834** |
| XGBoost MAPE | **13.38%** |
| MAE improvement vs 7-day seasonal naive | **35.1%** |
| RMSE improvement vs 7-day seasonal naive | **37.7%** |
| MAPE improvement vs 7-day seasonal naive | **32.5%** |
| Current stockout pairs | **44 / 150** |
| Critical inventory pairs | **59 / 150** |
| Recommended replenishment | **117,293.68 units** |
| Historical stockout rate | **15.76%** |
| Historical lost-sales exposure | **60,149,442.80** |
| Suppliers analyzed | **8** |
| Supplier risk classification | **4 LOW · 4 MEDIUM · 0 HIGH** |
| Simulated stockout reduction | **73.1%** |
| Simulated avoided lost-sales value | **7,024,229.03** |

> **Important:** the dataset is synthetic. Financial outputs and replenishment impact are deterministic planning estimates under explicit assumptions, not realized business results.

---

## Demand Forecasting

Demand is modeled at the **store × product** level using lagged, rolling, calendar, promotion, and discount features. Historical features are shifted so the model does not directly use the current day's demand when constructing predictors.

### Benchmark

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Naive 1-Day | 10.5188 | 15.3764 | 23.6574% |
| Seasonal Naive 7-Day | 8.9313 | 13.2975 | 19.8286% |
| **XGBoost** | **5.7937** | **8.2834** | **13.3766%** |

### Method

- **13 features:** lags, rolling statistics, calendar variables, promotions, and discounts
- Chronological train/test split
- Naive and seasonal-naive benchmarks
- Three sequential chronological 30-day walk-forward folds
- Recursive 30-day forecast
- SHAP feature importance

Known future promotions and discounts can be supplied through `data/forecast_scenario.csv`. Unspecified future days default to no promotion and zero discount.

> The headline improvement percentages are calculated on the current chronological holdout against the 7-day seasonal-naive benchmark. The three-fold walk-forward evaluation is a separate robustness check and evaluates one-step-ahead predictions.

---

## Inventory Policy

Forecasts feed a transparent replenishment policy:

```text
Forecast Demand
      ↓
Lead-Time Demand
      ↓
Safety Stock
      ↓
Reorder Point
      ↓
EOQ
      ↓
Recommended Order Quantity
```

Current classification across 150 store × product pairs:

- **59 CRITICAL**
- **48 REORDER**
- **43 NORMAL**

Additional signals:

- **44** current stockout pairs
- **93** pairs with less than 7 days of projected stock
- **117,293.68** recommended replenishment units

The implementation uses standard inventory-policy formulas and business rules. It is **not** a globally constrained optimization solver.

---

## Supplier Risk

Supplier performance is evaluated using operational KPIs:

- Average lead time
- On-time delivery
- Defect rate
- Supplier delays
- Fill rate
- Ordered vs received quantity

A transparent weighted supplier-risk score maps suppliers into **LOW / MEDIUM / HIGH** categories.

Current classification:

**4 LOW · 4 MEDIUM · 0 HIGH**

A separate temporal layer compares supplier × product behavior against a **14-day rolling baseline** and generates deterioration signals on a 0–100 scale.

These outputs are intended for **prioritization and monitoring**, not calibrated probabilities of future disruption.

---

## Financial Analytics & Simulation

The financial layer connects operational data to:

- Revenue and COGS
- Gross profit and margin
- Budget vs actual
- Inventory turnover
- Days Inventory Outstanding (DIO)
- Holding cost
- Historical lost-sales exposure
- Promotion effectiveness
- ABC product analysis
- Supplier-level stockout exposure

### 30-day replenishment scenario

The simulation compares the current recommended order policy against a **no-new-order baseline**.

| Output | Value |
|---|---:|
| Baseline projected shortage | **121,756.03 units** |
| Recommended-policy shortage | **32,775.05 units** |
| Stockout reduction | **88,980.98 units (73.1%)** |
| Avoided lost-sales value | **7,024,229.03** |
| Incremental holding cost | **54,117,463.62** |
| Estimated net benefit | **-47,093,234.59** |

The negative net benefit is intentionally shown: reducing stockouts does **not** automatically mean the policy is economically attractive under the current assumptions.

### Planning assumptions

- Unit cost is modeled as **60% of list price** because procurement cost is not included in the source data.
- Budget values are modeled planning assumptions rather than historical company budgets.
- The simulation is deterministic and uses forecast demand; it is not causal evidence of realized savings or ROI.

---

## Dashboard

![Dashboard Interface](images/Interface.png)

The Streamlit application provides nine views:

| View | Purpose |
|---|---|
| **Control Tower** | Prioritized actions with store and priority filters |
| **Financial Analytics** | Revenue, margin, DIO, holding cost, budget variance |
| **Sales & Revenue** | Promotion effectiveness and ABC analysis |
| **Inventory** | Stockouts, coverage, replenishment, simulation |
| **Supplier Risk** | Supplier performance and financial exposure |
| **Risk Signals** | Temporal supplier × product deterioration |
| **Forecast** | 30-day store × product forecasts and scenario inputs |
| **Model Benchmark** | Baseline, XGBoost, and walk-forward metrics |
| **Recommendations** | Action-oriented replenishment priorities |

The dashboard consumes generated CSV outputs; it does not require a live database connection at runtime.

---

## SQL Analytics

The PostgreSQL layer provides business-oriented analysis across sales, inventory, products, stores, and suppliers.

It demonstrates:

- Multi-table joins
- Aggregations and `GROUP BY`
- `CASE WHEN` business logic
- CTEs
- Window functions
- `LAG`
- Ranking and `PARTITION BY`
- Period-over-period analysis
- Revenue, demand, stockout, and supplier KPIs

SQL files:

```text
sql/01_business_analysis.sql
sql/02_finance_analysis.sql
```

---

## Reproducible Pipeline

The full workflow can be executed with:

```bash
python src/run_pipeline.py
```

The runner executes each stage sequentially and stops if a stage fails.

```text
baseline_forecasting.py
        ↓
sku_level_forecasting.py
        ↓
forecast_backtesting.py
        ↓
inventory_optimization.py
        ↓
supplier_risk.py
        ↓
temporal_disruption.py
        ↓
create_control_tower.py
        ↓
business_impact.py
        ↓
business_impact_simulation.py
        ↓
finance_analysis.py
        ↓
shap_explainability.py
        ↓
validate_outputs.py
```

`validate_outputs.py` checks generated artifacts and key output constraints before the pipeline is considered complete.

---

## Repository Structure

```text
.
├── app.py
├── data/
│   ├── retail_sales_data.csv
│   ├── forecast_scenario.csv
│   └── forecast_scenario_template.csv
├── images/
│   ├── banner.png
│   ├── Interface.png
│   ├── business/
│   └── forecasting/
├── sql/
│   ├── 01_business_analysis.sql
│   └── 02_finance_analysis.sql
├── src/
│   ├── generate_dataset.py
│   ├── load_to_postgres.py
│   ├── baseline_forecasting.py
│   ├── sku_level_forecasting.py
│   ├── forecast_backtesting.py
│   ├── inventory_optimization.py
│   ├── supplier_risk.py
│   ├── temporal_disruption.py
│   ├── create_control_tower.py
│   ├── business_impact.py
│   ├── business_impact_simulation.py
│   ├── finance_analysis.py
│   ├── shap_explainability.py
│   ├── validate_outputs.py
│   └── run_pipeline.py
├── .gitignore
├── requirements.txt
└── README.md
```

Generated pipeline outputs are ignored by Git, keeping the repository focused on source code and reproducible inputs.

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/Atul1127/ML-Driven-Supply-Chain-Control-Tower.git
cd ML-Driven-Supply-Chain-Control-Tower
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

**Windows Git Bash:**

```bash
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python src/run_pipeline.py
```

### 5. Launch the dashboard

```bash
streamlit run app.py
```

### Optional: PostgreSQL workflow

```bash
python src/load_to_postgres.py
```

PostgreSQL is used for the SQL analytics workflow. The Streamlit dashboard reads the generated CSV outputs directly.

---

## Dataset

The included dataset is synthetic and generated to contain realistic supply-chain patterns for experimentation:

- **5** retail stores
- **30** products across **6** categories
- **8** suppliers
- **2 years** of daily observations
- Demand seasonality and weekly effects
- Promotions and discounts
- Inventory and stockout events
- Supplier lead time, reliability, defects, and delays

The generator uses a fixed random seed for reproducibility.

---

## Limitations

This project is intentionally scoped as a portfolio/research prototype rather than a production supply-chain platform.

- Synthetic data is not representative of a real retailer.
- Forecast performance is specific to this dataset and evaluation setup.
- Walk-forward validation is one-step-ahead rather than a fully recursive multi-day backtest.
- Supplier risk is a transparent scoring framework, not a calibrated probability model.
- Inventory logic is policy-based rather than globally constrained optimization.
- Future promotions and discounts are supplied as scenario inputs rather than forecast by a separate promotion model.
- The replenishment simulation is deterministic and assumption-driven.
- There is no streaming ingestion, automated retraining, production orchestration, or live ERP integration.

---

## Tech Stack

**Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · PostgreSQL · psycopg2 · Streamlit**

---

## Project Focus

The goal is to demonstrate how **SQL analytics, machine learning, inventory policy, supplier monitoring, financial modeling, and dashboarding** can be combined into one actionable decision-support workflow.
