# ML-Driven Supply Chain Control Tower

![Supply Chain Control Tower](images/banner.png)

> An end-to-end supply-chain analytics system that turns retail and supplier data into **forecasts, inventory decisions, risk signals, financial exposure, and prioritized actions**.

**Python · XGBoost · SHAP · PostgreSQL · Streamlit · Supply Chain Analytics**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](#)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-EC6A00)](#)
[![PostgreSQL](https://img.shields.io/badge/SQL-PostgreSQL-4169E1?logo=postgresql&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](#)

## Overview

Supply-chain teams do not need another isolated forecasting model. They need a system that connects **demand → inventory → suppliers → financial impact → action**.

This project builds that workflow end to end on a synthetic retail dataset. The result is a Streamlit control tower backed by a reproducible analytics and ML pipeline.

### What the system answers

- **What is happening?** — sales, inventory, supplier, and financial KPIs
- **What is likely to happen?** — 30-day demand forecasts at store × product level
- **What needs attention?** — stockout, low-coverage, replenishment, and supplier-risk signals
- **What should we do?** — prioritized replenishment recommendations
- **What could it cost?** — lost-sales exposure, holding cost, and planning-simulation impact

---

## System Architecture

```text
                    Retail + Supplier Data
                              │
              ┌───────────────┴───────────────┐
              │                               │
        SQL Business Analytics          Data Preparation
              │                               │
              └───────────────┬───────────────┘
                              │
                    Demand Forecasting
                  XGBoost + Baselines
                              │
                   Chronological Validation
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        Inventory Policy   Supplier Risk   Financials
        SS / ROP / EOQ     Temporal Risk    KPIs + Exposure
              │               │               │
              └───────────────┼───────────────┘
                              │
                  Replenishment Simulation
                              │
                     Decision Support
                              │
                    Streamlit Control Tower
```

The project intentionally focuses on a **single coherent decision workflow** rather than collecting unrelated ML techniques.

---

## Results Snapshot

Current pipeline results:

| Metric | Result |
|---|---:|
| Daily records | **109,650** |
| Store × product pairs | **150** |
| Forecast MAE | **5.7937** |
| Forecast RMSE | **8.2834** |
| Forecast MAPE | **13.38%** |
| MAE improvement vs 7-day seasonal naive | **35.1%** |
| RMSE improvement vs 7-day seasonal naive | **37.7%** |
| MAPE improvement vs 7-day seasonal naive | **32.5%** |
| Current stockout pairs | **44 / 150** |
| Critical inventory pairs | **59 / 150** |
| Recommended replenishment | **117,293.68 units** |
| Historical stockout rate | **15.76%** |
| Historical lost-sales value | **60,149,442.80** |
| Suppliers analyzed | **8** |
| Supplier risk | **4 LOW · 4 MEDIUM · 0 HIGH** |
| Simulated stockout reduction | **73.1%** |
| Simulated avoided lost-sales value | **7,024,229.03** |

> **Important:** all data is synthetic. Financial results and replenishment impact are planning estimates based on explicit assumptions, not realized business results.

---

## Forecasting

Demand is forecast at the **store × product** level using leakage-safe time-series features.

### Model benchmark

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Naive 1-Day | 10.5188 | 15.3764 | 23.6574% |
| Seasonal Naive 7-Day | 8.9313 | 13.2975 | 19.8286% |
| **XGBoost** | **5.7937** | **8.2834** | **13.3766%** |

### Method

- 13 lag, rolling, calendar, promotion, and discount features
- Chronological train/test split
- Naive and seasonal-naive baselines
- Three chronological walk-forward validation windows
- Recursive 30-day forecast
- SHAP-based feature importance

Known future promotions and discounts can be supplied through `data/forecast_scenario.csv`.

> The headline improvement percentages above are calculated against the 7-day seasonal-naive model on the current chronological holdout. The walk-forward evaluation is a separate robustness check and is one-step-ahead in feature construction.

---

## Inventory Decisions

The forecast feeds a practical replenishment layer:

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
Replenishment Recommendation
```

The current run classifies the 150 store × product pairs as:

- **59 CRITICAL**
- **48 REORDER**
- **43 NORMAL**

It also identifies **44 current stockout pairs** and **93 pairs with less than 7 days of coverage**.

These are transparent inventory-policy calculations, not a globally constrained optimization solver.

---

## Supplier Risk

Supplier monitoring combines operational KPIs with temporal deterioration signals.

### Supplier performance

The system analyzes:

- lead time
- on-time delivery
- defect rate
- delays
- ordered vs received quantity
- fill rate
- rule-based supplier risk score

Current supplier classification:

**4 LOW · 4 MEDIUM · 0 HIGH**

### Temporal risk

A separate supplier × product layer compares recent behavior against a **14-day rolling baseline** and produces deterioration signals on a 0–100 scale.

The project intentionally treats these as **decision-support signals**, not calibrated disruption probabilities.

---

## Financial Impact & Simulation

The system connects operational decisions to commercial metrics:

- Revenue
- COGS
- Gross profit and margin
- Budget vs actual
- Inventory turnover
- Days Inventory Outstanding (DIO)
- Holding cost
- Historical lost-sales exposure
- Promotion effectiveness
- ABC analysis
- Supplier-level financial exposure

A 30-day simulation compares recommended replenishment with a no-new-order baseline.

| Simulation output | Value |
|---|---:|
| Baseline projected shortage | **121,756.03 units** |
| Recommended-policy projected shortage | **32,775.05 units** |
| Stockout reduction | **88,980.98 units (73.1%)** |
| Avoided lost-sales value | **7,024,229.03** |
| Incremental holding cost | **54,117,463.62** |
| Estimated net benefit | **-47,093,234.59** |

The negative estimated net benefit is deliberately reported. The simulation should be interpreted as a **planning scenario**, not proof of positive ROI.

### Key assumptions

- Unit cost = **60% of list price** because procurement cost is not present in the source dataset.
- Budget values are modeled planning assumptions, not historical company budgets.
- Future promotions/discounts are scenario inputs; unknown future events are not predicted by the current system.

---

## Dashboard

![Dashboard Interface](images/Interface.png)

The Streamlit control tower brings the workflow together in one interface.

| View | Purpose |
|---|---|
| **Control Tower** | Prioritized inventory actions and filters |
| **Financial Analytics** | Revenue, margin, DIO, holding cost, budget variance |
| **Sales & Revenue** | Promotion effectiveness and ABC analysis |
| **Inventory** | Stockouts, coverage, replenishment, policy simulation |
| **Supplier Risk** | Supplier performance and financial exposure |
| **Risk Signals** | Temporal supplier × product deterioration |
| **Forecast** | 30-day demand forecasts and scenario inputs |
| **Model Benchmark** | Baselines, XGBoost, and validation metrics |
| **Recommendations** | Prioritized actions and simulated impact |

---

## SQL Analytics

PostgreSQL is used for business-oriented analytics across sales, inventory, stores, products, and suppliers.

The SQL layer demonstrates:

- multi-table joins
- aggregations and `GROUP BY`
- `CASE WHEN` business logic
- CTEs
- window functions
- `LAG`
- ranking and partitioning
- period-over-period analysis
- revenue, demand, stockout, and supplier KPIs

Files:

```text
sql/01_business_analysis.sql
sql/02_finance_analysis.sql
```

---

## Reproducible Pipeline

Run everything with one command:

```bash
python src/run_pipeline.py
```

The pipeline executes the forecasting, inventory, supplier-risk, financial, simulation, explainability, and validation stages sequentially and stops if a stage fails.

Core stages:

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

The validation stage checks required outputs, benchmark metrics, backtest folds, forecast coverage, scenario bounds, finance reconciliation, risk ranges, and simulation results.

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

Generated pipeline outputs are ignored by Git to keep the repository focused on source code and reproducible inputs.

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

### Optional: PostgreSQL

```bash
python src/load_to_postgres.py
```

The PostgreSQL loader validates the dataset and populates the core tables used by the SQL workflow.

---

## Limitations & Scope

This project is deliberately transparent about what it does and does not claim.

- The dataset is synthetic: **5 stores, 30 products, 8 suppliers, 2023–2024 dates**.
- Forecast evaluation is chronological; walk-forward validation is one-step-ahead in feature construction rather than a fully recursive multi-day backtest.
- Supplier risk is a transparent prioritization framework, not a calibrated probability model.
- Inventory logic uses practical policy formulas rather than globally constrained optimization.
- Replenishment impact is a deterministic simulation under explicit assumptions.
- The project does not claim production deployment, streaming infrastructure, automated retraining, or real-world disruption prediction.

---

## Tech Stack

**Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · PostgreSQL · psycopg2 · Streamlit**

---

## Why This Project

This project is designed to demonstrate the combination of **data analytics, machine learning, SQL, optimization-style decision logic, and business communication** required to turn raw operational data into an actionable supply-chain decision system.
