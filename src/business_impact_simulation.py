"""Simulate the financial and service impact of forecast-driven replenishment.

The simulation compares a no-new-order baseline with the current recommended
order policy over the 30-day forecast horizon. It is explicitly a planning
simulation, not a claim of realized savings.
"""

from pathlib import Path
import numpy as np
import pandas as pd

RETAIL_PATH = Path("data/retail_sales_data.csv")
FORECAST_PATH = Path("data/sku_30_day_forecast.csv")
INVENTORY_PATH = Path("data/inventory_optimization_results.csv")
OUTPUT_PATH = Path("data/replenishment_impact_simulation.csv")
UNIT_COST_RATE = 0.60


def simulate():
    retail = pd.read_csv(RETAIL_PATH, parse_dates=["date"])
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    inventory = pd.read_csv(INVENTORY_PATH)

    latest = (retail.sort_values("date")
              .groupby(["store", "product"], as_index=False)
              .tail(1)[["store", "product", "closing_stock"]])
    params = inventory[["store", "product", "current_stock", "recommended_order_qty", "lead_time_days"]]
    params = params.merge(latest, on=["store", "product"], how="left", suffixes=("", "_latest"))
    params["start_stock"] = params["current_stock"].fillna(params["closing_stock"]).clip(lower=0)

    rows = []
    for r in params.itertuples(index=False):
        series = forecast[(forecast["store"] == r.store) & (forecast["product"] == r.product)].sort_values("date")
        if series.empty:
            continue
        demand = series.forecast_demand.to_numpy(dtype=float)
        lead_time = max(int(round(r.lead_time_days)), 0)
        order_qty = max(float(r.recommended_order_qty), 0)

        baseline_stock = float(r.start_stock)
        policy_stock = float(r.start_stock)
        baseline_shortage = 0.0
        policy_shortage = 0.0
        baseline_holding = 0.0
        policy_holding = 0.0
        receipts = {}

        for day, d in enumerate(demand):
            baseline_shortage += max(d - baseline_stock, 0)
            baseline_stock = max(baseline_stock - d, 0)
            baseline_holding += baseline_stock * UNIT_COST_RATE * float(series.iloc[day].unit_price)

            if day == 0 and order_qty > 0:
                receipts.setdefault(day + lead_time, 0.0)
                receipts[day + lead_time] += order_qty
            if day in receipts:
                policy_stock += receipts[day]
            policy_shortage += max(d - policy_stock, 0)
            policy_stock = max(policy_stock - d, 0)
            policy_holding += policy_stock * UNIT_COST_RATE * float(series.iloc[day].unit_price)

        lost_sales_reduction = baseline_shortage - policy_shortage
        incremental_inventory_cost = max(policy_holding - baseline_holding, 0)
        avg_price = float(series.unit_price.mean())
        avoided_lost_sales_value = lost_sales_reduction * avg_price
        net_benefit = avoided_lost_sales_value - incremental_inventory_cost

        rows.append({
            "store": r.store,
            "product": r.product,
            "baseline_stockout_units": round(baseline_shortage, 2),
            "recommended_policy_stockout_units": round(policy_shortage, 2),
            "stockout_reduction_units": round(lost_sales_reduction, 2),
            "recommended_order_qty": round(order_qty, 2),
            "avoided_lost_sales_value": round(avoided_lost_sales_value, 2),
            "incremental_holding_cost": round(incremental_inventory_cost, 2),
            "estimated_net_benefit": round(net_benefit, 2),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No forecast/inventory pairs were available for simulation.")
    result.to_csv(OUTPUT_PATH, index=False)

    summary = result[["baseline_stockout_units", "recommended_policy_stockout_units",
                      "stockout_reduction_units", "avoided_lost_sales_value",
                      "incremental_holding_cost", "estimated_net_benefit"]].sum()
    print("30-day replenishment policy simulation:")
    print(summary.round(2).to_string())
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    simulate()
