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
| XGBoost MAE / RMSE / WAPE | **5.78 / 8.29 / 12.7%** |
| MAE improvement vs 7-day seasonal naive | **35.2%** |
| MAE improvement vs promotion-adjusted seasonal naive (fair baseline) | **26.3%** |
| Recursive 30-day forecast MAE (backtest, the forecast that drives inventory) | **4.64** vs **8.33** naive |
| Current stockout pairs | **44 / 150** |
| Critical inventory pairs | **59 / 150** (all 59 already have an order in transit) |
| Recommended new replenishment | **73,627 units** across 52 pairs |
| Historical stockout rate | **15.76%** |
| Historical lost-sales exposure | **60.1M revenue · 22.9M gross profit** |
| Supplier risk classification | **4 LOW · 4 MEDIUM · 0 HIGH** |
| Simulated stockout reduction vs historical reorder rule | **70.2%** |
| Simulated 30-day net benefit vs historical rule | **+1.05M** |

> **Important:** the dataset is synthetic. Financial outputs and replenishment impact are deterministic planning estimates under explicit assumptions, not realized business results.

---

## Demand Forecasting

Demand is modeled at the **store × product** level using lagged, rolling, calendar, promotion, and discount features. Historical features are shifted so the model does not directly use the current day's demand when constructing predictors.

### Benchmark (60-day holdout, one day ahead)

| Model | MAE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|
| Naive 1-Day | 10.5188 | 15.3764 | 23.66% | 23.06% |
| Seasonal Naive 7-Day | 8.9310 | 13.2975 | 19.83% | 19.58% |
| Seasonal Naive 7-Day, promotion-adjusted | 7.8414 | 11.2486 | 17.68% | 17.19% |
| XGBoost without promotion features | 6.4490 | 9.7261 | 14.70% | 14.14% |
| **XGBoost** | **5.7830** | **8.2922** | **13.36%** | **12.68%** |

**Is the comparison fair?** XGBoost sees today's promotion flag and discount; plain naive forecasts do not. Two checks separate the two sources of gain:

- A **promotion-adjusted seasonal naive** (multiplies by the promotion uplift of 1.48×, learned on training data only). XGBoost still beats it by **26.3%** on MAE.
- An **XGBoost ablation without promotion features** still beats the plain seasonal naive by **27.8%**.

WAPE (total absolute error / total demand) is reported alongside MAPE because MAPE skips zero-demand days and over-weights slow items.

### Method

- **13 features:** lags, rolling statistics (computed on demand shifted by one day), calendar variables, promotions, and discounts
- Chronological train/test split; one global model across all 150 series
- Naive, seasonal-naive and promotion-adjusted benchmarks
- Three chronological 30-day walk-forward folds, evaluated two ways (below)
- Recursive 30-day planning forecast from a model retrained on all history
- SHAP feature importance (top drivers: 7- and 30-day rolling means, day of week, month, promotion)

### Backtesting the forecast that actually drives inventory

The holdout metrics are one day ahead (every prediction uses yesterday's real demand). The 30-day forecast used by inventory is **recursive**: day 2 uses the prediction for day 1 as its lag, and so on. `forecast_backtesting.py` backtests that exact procedure from three origins, with planned promotions supplied as known in advance:

| Evaluation (mean of 3 folds) | XGBoost MAE | Seasonal naive MAE |
|---|---:|---:|
| One step ahead | 4.56 | 7.32 |
| Recursive 30-day | 4.64 | 8.33 |
| Recursive, days 1–7 | 4.42 | 7.95 |
| Recursive, days 8–14 | 4.62 | 8.15 |
| Recursive, days 15–30 | 4.75 | 8.57 |

Error grows only slightly with horizon, so the planning forecast is close to the headline accuracy. Error by horizon is saved in `data/forecast_horizon_backtest.csv` and charted in the dashboard.

Known future promotions and discounts can be supplied through `data/forecast_scenario.csv`. The file in the repo is empty, so the current 30-day forecast assumes no promotions (about 5% of historical days are promotions, so this biases the forecast slightly down).

---

## Inventory Policy

Forecasts feed a transparent continuous-review, order-up-to policy:

```text
σ_e  = forecast-error RMSE per series (60-day holdout residuals)
L, σ_L = mean and std of REALIZED supplier lead time (orders matched to deliveries)
SS   = z · √(L·σ_e² + d²·σ_L²)          z = 1.645 (95% cycle service)
ROP  = d · L + SS
IP   = on hand + on order               (orders placed, not yet received)
Q    = min(EOQ, 30-day forecast)        EOQ holding cost = 25%/yr of unit COST
If IP ≤ ROP: order up to S = ROP + Q  →  qty = S − IP
```

Design choices:

- **Safety stock uses forecast error, not raw demand variability.** Raw demand std includes trend, seasonality and promotion spikes that the forecast already predicts. Forecast-error σ is about 28% lower (7.3 vs 10.1 units/day on average).
- **Lead-time uncertainty is included.** Late deliveries arrive 2 days late, so realized lead time varies (σ_L 0.4–0.7 days by supplier). For these items, the lead-time term is roughly as large as the forecast-error term, which is how supplier reliability enters the inventory maths.
- **Orders use inventory position.** 89 of 150 items have an order in transit. The previous rule ignored this and ordered again; it recommended 117,929 units, against 73,627 now.
- **Cycle stock is capped at 30 days of cover.** Under the 2,000-per-order assumption, uncapped EOQ is about 100 days of demand.

Current classification across 150 store × product pairs (status is based on stock on hand):

- **59 CRITICAL**: all already have an order in transit, so the action is to expedite it
- **49 REORDER**
- **42 NORMAL**

Control-tower priority: CRITICAL items are HIGH. They become **URGENT** when the supplier is HIGH risk, or when the item is out of stock with nothing in transit. Neither applies today, so URGENT is currently 0.

The implementation uses standard inventory-policy formulas and business rules. It is **not** a globally constrained optimization solver: there are no budget, space or minimum-order constraints.

---

## Supplier Risk

Supplier performance is evaluated using operational KPIs:

- Average lead time
- On-time delivery
- Defect rate
- Supplier delays
- Fill rate
- Ordered vs received quantity

A transparent weighted supplier-risk score maps suppliers into **LOW / MEDIUM / HIGH** categories. On-time rate, lead time and fill rate are measured from **realized purchase orders**: each order is matched to its delivery (one order in transit per item, so the k-th order pairs with the k-th receipt). The dataset's `on_time_rate` and `lead_time_days` columns are the generator's fixed settings and never change. Defect rate is the only input that still comes from a fixed configured value.

Current classification:

**4 LOW · 4 MEDIUM · 0 HIGH**

A separate temporal layer compares each supplier × product's last **28 days of deliveries** (realized lead time, on-time share, fill rate) and its stockout share against the **preceding 90 days**, and produces a 0–100 deterioration signal. Earlier versions computed a daily fill rate as same-day receipts ÷ same-day orders, which almost never coincide, so it was effectively always 0. They also used the constant setting columns, which flagged 56% of days as early warnings. Now 80% of days are NORMAL and 7% EMERGING or CRITICAL. Fill rate is 100% on every delivered order in this dataset, so it adds no signal. Thresholds are not calibrated, because the synthetic data has no labelled disruption episodes.

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

Each item is simulated day by day on forecast demand, starting from current stock **plus orders already in transit**, under three policies:

| Policy | Rule | Stockout units |
|---|---|---:|
| No new orders (reference only) | never order | 91,327 |
| **Historical rule (baseline)** | the reorder rule in the data: fixed `reorder_qty` when on-hand ≤ `reorder_point`, one order in transit | **31,961** |
| **Recommended policy** | order-up-to on inventory position, can reorder during the month | **9,537** |

| Recommended vs historical rule (30 days) | Value |
|---|---:|
| Stockout reduction | **22,424 units (70.2%)** |
| Avoided lost revenue | 1,947,145 |
| Avoided lost gross profit (40% margin) | **778,858** |
| Incremental holding cost (25%/yr, charged daily) | 48,031 |
| Ordering cost change (142 vs 303 orders × 2,000) | −322,000 |
| **Estimated net benefit** | **+1,052,827** |

> **Correction:** an earlier version charged the *full* cost value of inventory as holding cost every day, instead of 25%/365 of it. That is about 1,460× too much, and it produced a reported "net benefit" of −47M. It also compared against a no-order baseline and placed only one order per item. All three issues are fixed, and a unit test now guards the holding-cost formula.

### Planning assumptions

- Unit cost is modeled as **60% of list price** because procurement cost is not included in the source data.
- Holding cost is **25% of inventory cost value per year**; ordering cost is **2,000 per purchase order**.
- Lost sales are valued at gross profit for net benefit (revenue overstates the loss).
- There is no real budget in the data. "Budget vs actual" is **growth vs a target of last month +5%**, and the dashboard labels it that way.
- Inventory turnover is annual COGS ÷ average **total** inventory value per day: **38×**, a DIO of about **9.6 days**. An earlier version averaged inventory per row, which reported a 0.03-day DIO.
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

`validate_outputs.py` checks generated artifacts and key output constraints before the pipeline is considered complete. Among them: the holding-cost ceiling, that orders count in-transit stock, that DIO is in a sane range, and that the recursive backtest is present.

Shared assumptions, features, metrics and the order-matching helper live in `src/common.py`, so every step uses the same definitions.

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

`tests/test_formulas.py` covers the decision-layer formulas: daily holding cost, safety stock, EOQ on unit cost, order-up-to with in-transit stock, order-to-delivery matching, inventory turnover/DIO, and WAPE.

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
│   ├── common.py
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
├── tests/
│   ├── conftest.py
│   └── test_formulas.py
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
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
- Supplier settings (lead time, on-time probability, defect rate) are constant per supplier in the generator, so supplier behaviour does not drift over time and there are no disruption episodes to detect.
- The model forecasts true demand; real systems observe sales, which are censored by stockouts.
- No store or product ID feature; the global model infers each series' level from its lags.
- Supplier risk is a transparent scoring framework, not a calibrated probability model.
- Inventory logic is policy-based rather than globally constrained optimization.
- Future promotions and discounts are supplied as scenario inputs rather than forecast by a separate promotion model.
- The replenishment simulation is deterministic, uses forecast (not actual) demand and planned lead times, and is assumption-driven.
- There is no streaming ingestion, automated retraining, production orchestration, or live ERP integration.

---

## Tech Stack

**Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · PostgreSQL · psycopg2 · Streamlit**

---

## Project Focus

The goal is to demonstrate how **SQL analytics, machine learning, inventory policy, supplier monitoring, financial modeling, and dashboarding** can be combined into one actionable decision-support workflow.
