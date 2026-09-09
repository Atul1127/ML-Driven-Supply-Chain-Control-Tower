"""Post-run validation for generated control-tower artifacts."""

from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data")
EXPECTED_PAIRS = 150

REQUIRED = {
    "baseline_results.csv": ["model", "MAE", "RMSE", "MAPE", "sku_store_pairs"],
    "sku_xgboost_results.csv": ["model", "MAE", "RMSE", "MAPE", "sku_store_pairs"],
    "forecast_backtest_results.csv": ["fold", "model", "MAE", "RMSE", "MAPE"],
    "sku_30_day_forecast.csv": ["date", "store", "product", "forecast_demand", "promo_event", "discount_pct"],
    "inventory_optimization_results.csv": ["store", "product", "inventory_status"],
    "supplier_risk_analysis.csv": ["supplier", "risk_score", "risk_level"],
    "temporal_disruption.csv": ["date", "supplier", "product", "disruption_signal", "disruption_stage", "is_disruption"],
    "control_tower_inventory.csv": ["store", "product", "priority", "action"],
    "business_impact.csv": ["total_sku_store_pairs", "current_stockout_pairs", "historical_lost_sales_value"],
    "replenishment_impact_simulation.csv": ["store", "product", "baseline_stockout_units", "recommended_policy_stockout_units", "stockout_reduction_units", "estimated_net_benefit"],
    "finance_summary.csv": ["revenue", "cogs", "gross_profit", "gross_margin_pct", "inventory_turnover", "days_inventory_outstanding", "annual_holding_cost"],
    "budget_vs_actual.csv": ["month", "actual_revenue", "budget_revenue", "revenue_variance", "revenue_variance_pct", "actual_cogs", "budget_cogs"],
    "promotion_effectiveness.csv": ["promo_event", "days", "units_sold", "revenue", "conversion_pct"],
    "abc_analysis.csv": ["product", "category", "revenue", "revenue_share_pct", "cumulative_share_pct", "abc_class"],
    "supplier_stockout_impact.csv": ["supplier", "lost_sales_units", "lost_sales_value", "stockout_days"],
    "sku_xgboost_shap_importance.csv": ["feature", "mean_abs_shap"],
}


def main():
    errors = []
    frames = {}

    for filename, columns in REQUIRED.items():
        path = DATA / filename
        if not path.exists():
            errors.append(f"Missing output: {path}")
            continue
        df = pd.read_csv(path)
        frames[filename] = df
        missing = [c for c in columns if c not in df.columns]
        if missing:
            errors.append(f"{filename}: missing columns {missing}")
        if df.empty:
            errors.append(f"{filename}: file is empty")

    if not errors:
        baseline = frames["baseline_results.csv"].set_index("model")
        xgb = frames["sku_xgboost_results.csv"].iloc[0]
        seasonal = baseline.loc["SKU Seasonal-Naive-7-Day"]
        if not (xgb["MAE"] < seasonal["MAE"] and xgb["RMSE"] < seasonal["RMSE"] and xgb["MAPE"] < seasonal["MAPE"]):
            errors.append("XGBoost does not beat the seasonal-naive baseline on all three metrics")

        backtest = frames["forecast_backtest_results.csv"]
        if backtest["fold"].nunique() < 3:
            errors.append("Walk-forward backtest must contain at least three evaluation folds")
        summary = backtest.groupby("model")[["MAE", "RMSE", "MAPE"]].mean()
        if "SKU-Store XGBoost" not in summary.index or "Seasonal-Naive-7-Day" not in summary.index:
            errors.append("Walk-forward backtest is missing one of the benchmark models")
        elif summary.loc["SKU-Store XGBoost", "MAE"] >= summary.loc["Seasonal-Naive-7-Day", "MAE"]:
            errors.append("XGBoost does not improve average walk-forward MAE")

        pairs = int(xgb["sku_store_pairs"])
        if pairs != EXPECTED_PAIRS:
            errors.append(f"Expected {EXPECTED_PAIRS} store/SKU pairs, found {pairs}")
        forecast = frames["sku_30_day_forecast.csv"]
        forecast_pairs = forecast[["store", "product"]].drop_duplicates().shape[0]
        if forecast_pairs != pairs:
            errors.append(f"Forecast covers {forecast_pairs} store/SKU pairs; model reports {pairs}")
        if not forecast["discount_pct"].between(0, 100).all() or not forecast["promo_event"].isin([0, 1]).all():
            errors.append("Forecast scenario inputs must have discount 0-100 and promo event 0/1")

        temporal = frames["temporal_disruption.csv"]
        if not temporal["disruption_signal"].between(0, 100).all():
            errors.append("Temporal disruption signal must be between 0 and 100")

        finance = frames["finance_summary.csv"].iloc[0]
        if finance["revenue"] <= 0 or finance["cogs"] < 0:
            errors.append("Finance revenue must be positive and COGS cannot be negative")
        if not np.isclose(finance["gross_profit"], finance["revenue"] - finance["cogs"]):
            errors.append("Gross profit does not reconcile to revenue minus COGS")
        if not 0 <= finance["gross_margin_pct"] <= 100:
            errors.append("Gross margin percentage must be between 0 and 100")
        if finance["inventory_turnover"] < 0 or finance["days_inventory_outstanding"] < 0:
            errors.append("Inventory turnover and DIO must be non-negative")

        abc_classes = set(frames["abc_analysis.csv"]["abc_class"].dropna().unique())
        if not abc_classes.issubset({"A", "B", "C"}):
            errors.append("ABC analysis contains invalid classification values")

        impact = frames["business_impact.csv"].iloc[0]
        if int(impact["total_sku_store_pairs"]) != EXPECTED_PAIRS:
            errors.append("Business impact pair count does not match expected coverage")
        if impact["historical_lost_sales_value"] < 0 or impact["current_inventory_value"] < 0:
            errors.append("Business impact financial values cannot be negative")

        simulation = frames["replenishment_impact_simulation.csv"]
        if (simulation["stockout_reduction_units"] < -1e-9).any():
            errors.append("Recommended replenishment should not increase simulated stockout units")

    if errors:
        print("OUTPUT VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("OUTPUT VALIDATION PASSED")
    print("- Required output files and schemas are present")
    print("- XGBoost beats the seasonal-naive baseline on the original holdout")
    print("- XGBoost improves average MAE across three walk-forward folds")
    print(f"- Forecast coverage matches the {EXPECTED_PAIRS} store/SKU pairs")
    print("- Future promotion/discount inputs are bounded and explicit")
    print("- Finance KPIs reconcile and fall within valid ranges")
    print("- Replenishment impact simulation is valid")
    print("- Temporal disruption signals are valid")


if __name__ == "__main__":
    main()
