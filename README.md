# AI-Driven Supply Chain Control Tower

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

## 📊 Model & Validation

The forecasting layer operates at the **store × product** level using leakage-safe lag and rolling features.

| Component | Approach |
|---|---|
| Naive benchmark | Previous-day demand |
| Seasonal benchmark | 7-day seasonal naive |
| Main model | XGBoost Regressor |
| Features | Lags, rolling statistics, calendar, promotion, discount |
| Primary evaluation | Chronological holdout |
| Robustness check | 3 chronological walk-forward windows |
| Metrics | MAE, RMSE, MAPE |
| Explainability | SHAP feature importance |
| Forecast horizon | 30 days, recursive |

The final validation script checks the required artifacts, benchmark performance, three backtest folds, forecast coverage, scenario bounds, finance reconciliation, temporal-risk ranges, and replenishment simulation outputs.

The project currently covers **150 store × product combinations** from a synthetic retail dataset containing **109,650 daily records**.

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

The inventory layer identifies **CRITICAL, REORDER, and NORMAL** states using forecast-driven lead-time demand, a 95% service-level safety-stock assumption, reorder point logic, EOQ, and a 30-day forecast cap.

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

A separate temporal layer compares supplier × product behavior against a preceding **14-day rolling baseline** and produces deterioration signals on a 0–100 scale.

The project intentionally avoids claiming calibrated disruption probabilities.

### 5. Business Impact Simulation

Recommended replenishment is compared with a **no-new-order baseline** over the 30-day forecast horizon.

The simulation estimates:

- Baseline projected shortage units
- Recommended-policy projected shortage units
- Stockout reduction
- Avoided lost-sales value
- Incremental holding cost
- Estimated net benefit

These are deterministic, synthetic planning estimates — **not realized savings or causal business impact**.

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
git clone https://github.com/Atul1127/AI-Driven-Supply-Chain-Control-Tower.git
cd AI-Driven-Supply-Chain-Control-Tower
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
- Walk-forward validation is one-step-ahead in feature construction rather than a fully recursive multi-day backtest.
- Future promotions/discounts are scenario inputs; unknown future events are not predicted by the current system.
- Supplier risk and temporal disruption are prioritization signals, not calibrated probabilities.
- Inventory formulas are practical policy heuristics rather than a globally constrained optimization model.
- Unit cost uses the documented **60%-of-list-price** assumption.
- Budget values are modeled planning assumptions rather than historical company budgets.
- Replenishment impact is a deterministic planning simulation and must not be presented as realized savings.
- The project does not claim production deployment, streaming infrastructure, automated retraining, or real-world disruption prediction.

## 🛠️ Technology Stack

**Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · PostgreSQL · psycopg2 · Streamlit**

The dependency list is intentionally limited to libraries used by the current project; unused visualization dependencies were removed during the final code-depth cleanup.
