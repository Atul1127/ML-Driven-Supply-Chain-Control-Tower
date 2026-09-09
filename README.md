# AI-Driven Supply Chain Control Tower

![Supply Chain Control Tower](images/banner.png)

**SQL Analytics · Demand Forecasting · Inventory Decisions · Supplier Risk · Financial Impact · Decision Support**

An end-to-end supply-chain analytics platform that turns retail sales and supplier operations into **business KPIs, demand forecasts, replenishment decisions, supplier-risk signals, financial exposure, and prioritized actions**.

> **Portfolio disclosure:** The dataset is synthetic. Financial outputs use explicit planning assumptions, and risk/disruption signals are decision-support analysis rather than calibrated probabilities or proven real-world ground truth.

## 🎯 What I Built

```text
Retail & Supplier Data
        ↓
SQL Business Analysis
        ↓
Demand Forecasting + Walk-Forward Validation
        ↓
Forecast-Driven Inventory Decisions
        ↓
Supplier Performance + Temporal Risk
        ↓
Business Impact Simulation
        ↓
Interactive Streamlit Control Tower
```

The focus is deliberately on a coherent decision workflow rather than collecting many unrelated ML algorithms.

## 📊 Current Results

Results from the reproducible pipeline run:

| Metric | Result |
|---|---:|
| Store × SKU combinations | **150** |
| Daily records | **109,650** |
| XGBoost MAE | **5.79** |
| Seasonal-naive MAE | **8.93** |
| Forecast MAE improvement | **35.1%** |
| Current stockout pairs | **44 / 150** |
| Low-coverage pairs (<7 days) | **93 / 150** |
| Historical lost-sales exposure | **₹6.21 Cr** |
| Current inventory value | **₹29.4 Lakh** |
| Recommended replenishment | **117,294 units** |

Financial values are derived from the synthetic dataset and documented assumptions; they are **not real company results**. Replenishment impact is a planning simulation, not realized savings.

## 🖥️ Dashboard

The Streamlit dashboard provides an executive view of operational and financial decisions.

![Dashboard Interface](images/Interface.png)

### Dashboard views

- 🚨 **Control Tower** — prioritized inventory actions and filters
- 💰 **Financial Analytics** — revenue, COGS, gross margin, budget variance, DIO and holding cost
- 📈 **Sales & Revenue** — promotion effectiveness and ABC product analysis
- 📦 **Inventory** — stockouts, inventory value, coverage, replenishment and policy simulation
- 🏭 **Supplier Risk** — supplier service performance and financial exposure
- ⚠️ **Risk Signals** — temporal supplier × product deterioration against a rolling baseline
- 🔮 **Forecast** — 30-day SKU/store demand forecasts with known future promotion inputs
- 📊 **Model Benchmark** — baseline versus XGBoost plus walk-forward backtesting
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

Files:

- `sql/01_business_analysis.sql`
- `sql/02_finance_analysis.sql`

### 2. Demand Forecasting

- Store × SKU daily demand aggregation
- Leakage-safe lag and rolling features
- 1-day naive and 7-day seasonal-naive baselines
- XGBoost forecasting
- Chronological evaluation
- **Three-window walk-forward backtesting**
- 30-day recursive forecasting
- MAE, RMSE and MAPE comparison
- SHAP feature importance

Known future promotion and discount inputs can be supplied through `data/forecast_scenario.csv`. Unspecified future days default to no promotion and zero discount.

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

The output identifies **CRITICAL, REORDER, and NORMAL** inventory states. Service level, ordering cost and holding-rate assumptions are explicit planning parameters.

### 4. Supplier Risk

Supplier performance combines:

- Lead time
- On-time delivery
- Defect rate
- Delays
- Ordered vs received quantity
- Fill rate
- Transparent supplier risk scoring

A separate temporal layer compares supplier × product behavior against a preceding **14-day baseline**, producing deterioration signals for prioritization.

The project intentionally does not claim calibrated disruption probabilities.

### 5. Business Impact Simulation

Recommended replenishment is tested over the 30-day forecast horizon against a no-new-order baseline.

The simulation estimates:

- Baseline projected shortage units
- Recommended-policy projected shortage units
- Stockout reduction
- Avoided lost-sales value
- Incremental holding cost
- Estimated net benefit

These are **synthetic planning estimates**, not realized financial savings.

### 6. Financial Analysis

The finance layer connects operational performance to commercial metrics:

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

**Budget assumption:** monthly budget is modeled from prior-month actual performance using explicit growth assumptions; it is not a historical company budget.

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

The pipeline intentionally removes the previous multi-algorithm clustering/PCA disruption experiment. Supplier risk is now based on transparent KPIs plus temporal deterioration signals, which are more appropriate for the small synthetic supplier population.

Validation checks required artifacts, forecast coverage, original benchmark performance, walk-forward performance, scenario bounds, finance reconciliation, and replenishment simulation outputs.

## 📁 Repository Structure

```text
.
├── app.py
├── data/
│   └── retail_sales_data.csv
├── images/
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

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Atul1127/AI-Driven-Supply-Chain-Control-Tower.git
cd AI-Driven-Supply-Chain-Control-Tower
```

### 2. Create and activate a virtual environment

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

### 4. Run the complete analytics pipeline

```bash
python src/run_pipeline.py
```

### 5. Launch the dashboard

```bash
streamlit run app.py
```

PostgreSQL is required for the database-loading/SQL workflow; the dashboard reads the generated CSV outputs.

## ⚠️ Evaluation & Limitations

- Forecasting uses chronological evaluation and three walk-forward windows; the original single holdout is retained for continuity with the headline benchmark.
- Future promotions/discounts are scenario inputs; unknown future events are not magically predicted.
- Supplier risk and temporal disruption are prioritization signals, not calibrated disruption probabilities.
- The dataset is synthetic: 5 stores, 30 products, 8 suppliers and 2023–2024 dates.
- Unit cost uses the documented **60%-of-list-price** assumption.
- Budget values are modeled planning assumptions rather than historical company budgets.
- Inventory formulas are simplified portfolio-level policies rather than a full constrained optimization solver.
- Replenishment impact is a deterministic planning simulation and should not be presented as realized savings.
- The project does not claim production deployment, streaming infrastructure, automated retraining, or real-world disruption prediction.
