# ML-Driven Supply Chain Control Tower

![Supply Chain Control Tower](images/banner.png)

**SQL Analytics · Demand Forecasting · Inventory Decisions · Supplier Risk · Financial Impact · Decision Support**

An end-to-end supply-chain analytics project that converts retail sales and supplier operations into **business KPIs, demand forecasts, replenishment decisions, supplier-risk signals, financial exposure, and prioritized actions**.

> **Portfolio disclosure:** The dataset is synthetic. Financial outputs use explicit planning assumptions, and supplier-risk/disruption signals are decision-support indicators rather than calibrated probabilities or real-world ground truth.

## 🎯 What I Built

```text
Retail & Supplier Data
        ↓
SQL Business Analytics
        ↓
Demand Forecasting
(XGBoost + Seasonal Naive)
        ↓
Chronological Walk-Forward Validation
        ↓
Inventory Decisions
(Safety Stock + ROP + EOQ)
        ↓
Supplier Performance + Temporal Risk
        ↓
Business Impact / Replenishment Simulation
        ↓
Interactive Streamlit Control Tower
```

The project deliberately focuses on one coherent decision workflow instead of collecting unrelated ML algorithms.

## 📊 Results Snapshot

The current pipeline run covers **109,650 daily records** and **150 store × product combinations**.

| Area | Result |
|---|---|
| Dataset scale | **109,650 daily records** |
| SKU-store coverage | **150 pairs** |
| XGBoost MAE | **5.7937** |
| XGBoost RMSE | **8.2834** |
| XGBoost MAPE | **13.3766%** |
| MAE improvement vs 7-day seasonal-naive | **35.1% lower** |
| RMSE improvement vs 7-day seasonal-naive | **37.7% lower** |
| MAPE improvement vs 7-day seasonal-naive | **32.5% lower** |
| Current stockout pairs | **44 / 150** |
| Low-coverage pairs (<7 days) | **93 / 150** |
| Critical inventory pairs | **59** |
| Recommended replenishment | **117,293.68 units** |
| Historical stockout rate | **15.76%** |
| Historical lost-sales value | **60,149,442.80** |
| Supplier records analyzed | **109,650** |
| Suppliers analyzed | **8** |
| Supplier risk | **4 LOW · 4 MEDIUM · 0 HIGH** |
| 30-day simulated stockout reduction | **88,980.98 units (73.1%)** |
| Simulated avoided lost-sales value | **7,024,229.03** |

> **Simulation disclosure:** The 30-day replenishment figures are deterministic planning estimates. The simulation produced **54,117,463.62** incremental holding cost and **-47,093,234.59** estimated net benefit, so the project does **not** claim realized savings or positive ROI.

## 📈 Forecasting Benchmark

The forecasting layer operates at the **store × product** level using leakage-safe lag and rolling features.

| Model | MAE | RMSE | MAPE | SKU-store pairs |
|---|---:|---:|---:|---:|
| Naive-1-Day | 10.5188 | 15.3764 | 23.6574% | 150 |
| Seasonal-Naive-7-Day | 8.9313 | 13.2975 | 19.8286% | 150 |
| **SKU-Store XGBoost** | **5.7937** | **8.2834** | **13.3766%** | **150** |

The XGBoost model achieved **35.1% lower MAE, 37.7% lower RMSE, and 32.5% lower MAPE** than the 7-day seasonal-naive benchmark on the current chronological holdout.

| Component | Approach |
|---|---|
| Naive benchmark | Previous-day demand |
| Seasonal benchmark | 7-day seasonal naive |
| Main model | XGBoost Regressor |
| Features | **13** lag, rolling, calendar, promotion and discount features |
| Primary evaluation | Chronological holdout |
| Robustness check | 3 chronological walk-forward windows |
| Metrics | MAE, RMSE, MAPE |
| Explainability | SHAP feature importance |
| Forecast horizon | 30 days, recursive |

The final validation script checks the required artifacts, benchmark performance, three backtest folds, forecast coverage, scenario bounds, finance reconciliation, temporal-risk ranges, and replenishment simulation outputs.

## 🖥️ Dashboard

![Dashboard Interface](images/Interface.png)

The Streamlit control tower provides:

- 🚨 **Control Tower** — prioritized inventory actions and store/priority filters
- 💰 **Financial Analytics** — revenue, COGS, gross margin, budget variance, DIO and holding cost
- 📈 **Sales & Revenue** — promotion effectiveness and ABC product analysis
- 📦 **Inventory** — stockouts, coverage, replenishment and policy simulation
- 🏭 **Supplier Risk** — supplier service performance and financial exposure
- ⚠️ **Risk Signals** — temporal supplier × product deterioration against a rolling baseline
- 🔮 **Forecast** — 30-day store × product demand forecasts and scenario inputs
- 📊 **Model Benchmark** — baseline versus XGBoost and walk-forward validation
- 🎯 **Recommendations** — prioritized actions and simulated policy benefit

## 🧩 Core Analytics

### 1. SQL Business Analytics

PostgreSQL analysis demonstrates practical analyst SQL:

- Multi-table `JOIN`s
- `GROUP BY` and aggregations
- `CASE WHEN` business rules
- CTEs
- Window functions
- `LAG` and period-over-period analysis
- `RANK` / `PARTITION BY`
- Revenue, demand, stockout and lost-sales KPIs
- Product, store and supplier performance

SQL files:

- `sql/01_business_analysis.sql`
- `sql/02_finance_analysis.sql`

### 2. Demand Forecasting

The forecasting pipeline:

1. Aggregates demand to store × product × day.
2. Builds leakage-safe lag and rolling features.
3. Evaluates naive and seasonal-naive benchmarks.
4. Trains XGBoost using a chronological split.
5. Runs three chronological walk-forward evaluation windows.
6. Generates a recursive 30-day forecast.
7. Produces SHAP feature importance.

The current XGBoost run uses **13 features**, **96,150 training rows**, **9,000 test rows**, and **150 SKU-store pairs**, achieving **MAE 5.7937, RMSE 8.2834, and MAPE 13.3766%**.

Known future promotions and discounts can be supplied through `data/forecast_scenario.csv`. Unspecified future days default to no promotion and zero discount.

> **Validation note:** the walk-forward evaluation is chronological and one-step-ahead in its feature construction; it should not be described as three fully recursive 30-day backtests.

### 3. Inventory Decisions

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
Recommended Replenishment
```

The inventory layer currently classifies the **150 SKU-store pairs** into **59 CRITICAL, 48 REORDER, and 43 NORMAL** states. It identifies **44 current stockout pairs**, **93 pairs with <7 days of coverage**, and a **15.76% historical stockout rate**.

The current run recommends **117,293.68 replenishment units**.

These are practical inventory-policy calculations rather than a full constrained optimization solver.

### 4. Supplier Risk

Supplier performance combines:

- Average lead time
- On-time delivery
- Defect rate
- Supplier delays
- Ordered vs received quantity
- Fill rate
- Transparent rule-based supplier risk score

The current run analyzes **109,650 supplier records across 8 suppliers**, with **4 LOW, 4 MEDIUM, and 0 HIGH** supplier-risk classifications.

A separate temporal layer compares supplier × product behavior against a preceding **14-day rolling baseline** and produces deterioration signals on a 0–100 scale.

The project intentionally avoids claiming calibrated disruption probabilities.

### 5. Business Impact Simulation

Recommended replenishment is compared with a **no-new-order baseline** over the 30-day forecast horizon.

The current simulation estimates:

- **121,756.03** baseline projected shortage units
- **32,775.05** recommended-policy projected shortage units
- **88,980.98 units (73.1%)** stockout reduction
- **7,024,229.03** avoided lost-sales value
- **54,117,463.62** incremental holding cost
- **-47,093,234.59** estimated net benefit

These are deterministic, synthetic planning estimates — **not realized savings or causal business impact**. The negative estimated net benefit is intentionally reported rather than presenting the avoided lost-sales value as profit or ROI.

### 6. Financial Analysis

The finance layer connects operational data to commercial metrics:

- Revenue
- COGS
- Gross Profit
- Gross Margin %
- Budget vs Actual
- Inventory Turnover
- Days Inventory Outstanding (DIO)
- Annual inventory holding cost
- Historical lost-sales exposure
- Promotion effectiveness
- ABC product classification
- Supplier-level stockout financial impact

The current business-impact analysis reports **60,149,442.80** in historical lost-sales value and **1,765,075.20** in current inventory value.

**Cost assumption:** unit cost is modeled as **60% of list price** because procurement cost is not available in the original synthetic dataset.

**Budget assumption:** monthly budgets are modeled from prior-month actual performance using explicit growth assumptions; they are not historical company budgets.

## 🔄 Reproducible Pipeline

```text
1. baseline_forecasting.py
2. sku_level_forecasting.py
3. forecast_backtesting.py
4. inventory_optimization.py
5. supplier_risk.py
6. temporal_disruption.py
7. create_control_tower.py
8. business_impact.py
9. business_impact_simulation.py
10. finance_analysis.py
11. shap_explainability.py
12. validate_outputs.py
```

`run_pipeline.py` executes the complete sequence and stops if any stage fails.

The earlier multi-algorithm clustering/PCA disruption experiment was intentionally removed. The current design uses transparent supplier KPIs plus temporal deterioration signals, which is more defensible for the small synthetic supplier population.

## 📁 Repository Structure

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

Generated pipeline CSV outputs are ignored by Git except for the source retail dataset and scenario files, keeping the repository focused on reproducible code rather than generated artifacts.

## 🚀 Quick Start

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

### 4. Run the complete pipeline

```bash
python src/run_pipeline.py
```

### 5. Launch the dashboard

```bash
streamlit run app.py
```

PostgreSQL is required for the database-loading and SQL workflow. The Streamlit dashboard reads the generated CSV outputs.

### Optional PostgreSQL workflow

```bash
python src/load_to_postgres.py
```

The loader validates the dataset, populates the `stores`, `suppliers`, `products`, `sales`, and `inventory` tables, and verifies record counts.

## ⚠️ Evaluation & Limitations

- The dataset is synthetic: 5 stores, 30 products, 8 suppliers and 2023–2024 dates.
- Forecasting uses chronological evaluation plus three walk-forward windows.
- The headline **35.1% MAE / 37.7% RMSE / 32.5% MAPE improvements** compare the current XGBoost chronological holdout with the 7-day seasonal-naive benchmark; they are not the three-fold walk-forward averages.
- Walk-forward validation is one-step-ahead in feature construction rather than a fully recursive multi-day backtest.
- Future promotions/discounts are scenario inputs; unknown future events are not predicted by the current system.
- Supplier risk and temporal disruption are prioritization signals, not calibrated probabilities.
- Inventory formulas are practical policy heuristics rather than a globally constrained optimization model.
- Unit cost uses the documented **60%-of-list-price** assumption.
- Budget values are modeled planning assumptions rather than historical company budgets.
- Replenishment impact is a deterministic planning simulation and must not be presented as realized savings; the current simulation produces a **negative estimated net benefit**.
- The project does not claim production deployment, streaming infrastructure, automated retraining, or real-world disruption prediction.

## 🛠️ Technology Stack

**Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · PostgreSQL · psycopg2 · Streamlit**

The dependency list is intentionally limited to libraries used by the current project; unused visualization dependencies were removed during the final code-depth cleanup.
