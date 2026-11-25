"""Forecast-driven SKU/store inventory policy.

Policy (continuous review, order-up-to):
- sigma_e   : daily forecast-error RMSE per series from the 60-day holdout
              (not the raw demand std, which double-counts trend/seasonality/promos)
- L, sigma_L: mean and std of REALIZED supplier lead time, from matched orders
- SS        = z * sqrt(L * sigma_e^2 + d^2 * sigma_L^2)
- ROP       = d * L + SS
- IP        = on hand + on order (orders placed but not yet received)
- EOQ       = sqrt(2 * D * S / H), H = 25% x unit COST per year
- Cycle stock Q = min(EOQ, 30-day forecast) (EOQ alone is ~100 days of cover
  under the 2,000-per-order assumption, so a 30-day cover cap is applied)
- If IP <= ROP: order up to S = ROP + Q, i.e. qty = S - IP

Status is based on stock on hand (what can be sold today); the order quantity
is based on inventory position, so items with an order already in transit are
not ordered twice.
"""

import numpy as np
import pandas as pd

from common import eoq, order_history, safety_stock, unit_cost

ORDER_CAP_DAYS = 30
RETAIL_PATH = "data/retail_sales_data.csv"
FORECAST_PATH = "data/sku_30_day_forecast.csv"
TEST_PRED_PATH = "data/sku_test_predictions.csv"
OUTPUT_PATH = "data/inventory_optimization_results.csv"


def main():
    retail = pd.read_csv(RETAIL_PATH, parse_dates=["date"])
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    holdout = pd.read_csv(TEST_PRED_PATH, parse_dates=["date"])
    orders = order_history(retail)

    days = max(int((retail["date"].max() - retail["date"].min()).days) + 1, 1)
    inv = retail.sort_values("date").groupby(["store", "product"], as_index=False).agg(
        category=("category", "first"),
        supplier=("supplier", "first"),
        total_demand=("demand", "sum"),
        unit_price=("unit_price", "mean"),
        planned_lead_time_days=("lead_time_days", "mean"),
        current_stock=("closing_stock", "last"),
        raw_demand_std=("demand", "std"),
        historical_stockout_days=("stockout", "sum"),
        observed_days=("date", "count"),
    )
    inv["annual_demand"] = inv["total_demand"] * (365.0 / days)
    inv["historical_stockout_rate"] = inv["historical_stockout_days"] / inv["observed_days"].clip(lower=1)

    # Forecast error per series (holdout residual RMSE).
    holdout["err2"] = (holdout["demand"] - holdout["predicted_demand"]) ** 2
    err = holdout.groupby(["store", "product"], as_index=False).agg(forecast_error_std=("err2", "mean"))
    err["forecast_error_std"] = np.sqrt(err["forecast_error_std"])

    # Realized lead time per supplier from matched purchase orders.
    closed = orders[~orders["open"]]
    lt = closed.groupby("supplier", as_index=False).agg(
        lead_time_days=("realized_lead_time", "mean"), lead_time_std=("realized_lead_time", "std"))

    # Stock already on order (in transit).
    on_order = orders[orders["open"]].groupby(["store", "product"], as_index=False).agg(on_order_qty=("ordered_qty", "sum"))

    fc = forecast.groupby(["store", "product"], as_index=False).agg(
        forecast_daily_demand=("forecast_demand", "mean"),
        forecast_30_day_demand=("forecast_demand", "sum"),
    )
    r = (inv.merge(fc, on=["store", "product"], how="left")
            .merge(err, on=["store", "product"], how="left")
            .merge(lt, on="supplier", how="left")
            .merge(on_order, on=["store", "product"], how="left"))
    r["forecast_daily_demand"] = r["forecast_daily_demand"].fillna(r["annual_demand"] / 365)
    r["forecast_30_day_demand"] = r["forecast_30_day_demand"].fillna(r["forecast_daily_demand"] * 30)
    r["forecast_error_std"] = r["forecast_error_std"].fillna(r["forecast_error_std"].median())
    r["lead_time_days"] = r["lead_time_days"].fillna(r["planned_lead_time_days"])
    r["lead_time_std"] = r["lead_time_std"].fillna(0)
    r["on_order_qty"] = r["on_order_qty"].fillna(0)
    r["inventory_position"] = r["current_stock"] + r["on_order_qty"]

    r["safety_stock"] = safety_stock(r["forecast_error_std"], r["lead_time_days"],
                                     r["forecast_daily_demand"], r["lead_time_std"])
    r["expected_lead_time_demand"] = r["forecast_daily_demand"] * r["lead_time_days"]
    r["reorder_point"] = r["expected_lead_time_demand"] + r["safety_stock"]
    r["unit_cost"] = unit_cost(r["unit_price"])
    r["eoq"] = eoq(r["annual_demand"], r["unit_price"])
    r["cycle_stock"] = np.minimum(r["eoq"], r["forecast_30_day_demand"] * ORDER_CAP_DAYS / 30)
    r["order_up_to_level"] = r["reorder_point"] + r["cycle_stock"]

    needs_order = r["inventory_position"] <= r["reorder_point"]
    r["recommended_order_qty"] = np.where(needs_order, (r["order_up_to_level"] - r["inventory_position"]).clip(lower=0), 0.0)
    r["inventory_status"] = np.select(
        [r.current_stock <= r.safety_stock, r.current_stock <= r.reorder_point],
        ["CRITICAL", "REORDER"], default="NORMAL")
    r["shortage_to_rop"] = np.maximum(r["reorder_point"] - r["inventory_position"], 0)
    r["days_of_stock"] = r["current_stock"] / r["forecast_daily_demand"].clip(lower=0.01)
    r["current_stockout"] = (r["current_stock"] <= 0).astype(int)
    r["low_coverage_7d"] = (r["days_of_stock"] < 7).astype(int)

    for col in ["annual_demand", "forecast_daily_demand", "forecast_30_day_demand", "raw_demand_std",
                "forecast_error_std", "lead_time_days", "lead_time_std", "safety_stock",
                "expected_lead_time_demand", "reorder_point", "eoq", "cycle_stock", "order_up_to_level",
                "recommended_order_qty", "shortage_to_rop", "days_of_stock", "historical_stockout_rate", "unit_cost"]:
        r[col] = r[col].round(2)

    r = r.sort_values(["inventory_status", "shortage_to_rop"], ascending=[True, False])
    r.to_csv(OUTPUT_PATH, index=False)
    print(r["inventory_status"].value_counts().to_string())
    print(f"Items with an order in transit: {int((r['on_order_qty'] > 0).sum())}/{len(r)}")
    print(f"Items needing a new order: {int((r['recommended_order_qty'] > 0).sum())}/{len(r)}")
    print(f"Recommended units: {r['recommended_order_qty'].sum():,.0f}")
    print(f"Mean safety stock: {r['safety_stock'].mean():.1f} units "
          f"(raw-demand-std method would give {(1.645 * r['raw_demand_std'] * np.sqrt(r['lead_time_days'])).mean():.1f})")
    print(f"Current stockouts: {int(r['current_stockout'].sum())}/{len(r)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
