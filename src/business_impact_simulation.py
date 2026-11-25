"""Simulate the financial and service impact of forecast-driven replenishment.

Each store/product is simulated day by day over the 30-day forecast horizon
under three policies, all starting from current stock plus orders already in
transit:

- no_new_orders : reference only (almost anything beats never ordering)
- historical    : the replenishment rule already in the data (order the fixed
                  reorder_qty when on-hand stock <= reorder_point, one order
                  in transit at a time). This is the baseline we compare against.
- recommended   : the control-tower policy (continuous review on inventory
                  position, order up to S when IP <= ROP; can reorder during
                  the month).

Costs
- holding cost per day = closing stock x unit cost x (25% / 365). The previous
  version charged the full cost value every day (~1,460x too high).
- ordering cost = 2,000 per purchase order.
- lost sales are valued at revenue AND at gross profit (40% of price under the
  60% unit-cost assumption); net benefit uses gross profit.

This is a deterministic planning simulation on forecast demand, not a claim
of realized savings.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from common import ORDERING_COST, UNIT_COST_RATE, daily_holding_cost, order_history

RETAIL_PATH = Path("data/retail_sales_data.csv")
FORECAST_PATH = Path("data/sku_30_day_forecast.csv")
INVENTORY_PATH = Path("data/inventory_optimization_results.csv")
OUTPUT_PATH = Path("data/replenishment_impact_simulation.csv")


def simulate_policy(demand, start_stock, arrivals, lead_time, unit_price, rule="none",
                    reorder_point=0.0, order_up_to=0.0, fixed_qty=0.0):
    """Run one policy. arrivals: {day_index: qty} for orders already in transit.

    rule: "none" | "fixed_qty_on_hand" (historical) | "order_up_to_ip" (recommended)
    Day loop: receive -> sell -> review/order -> pay holding on closing stock.
    """
    stock = float(start_stock)
    pipeline = {int(k): float(v) for k, v in arrivals.items()}
    shortage = holding = 0.0
    orders = 0
    units_ordered = 0.0
    for day, d in enumerate(demand):
        stock += pipeline.pop(day, 0.0)
        shortage += max(d - stock, 0.0)
        stock = max(stock - d, 0.0)
        on_order = sum(q for t, q in pipeline.items() if t > day)
        qty = 0.0
        if rule == "fixed_qty_on_hand" and stock <= reorder_point and on_order == 0:
            qty = fixed_qty
        elif rule == "order_up_to_ip" and stock + on_order <= reorder_point:
            qty = max(order_up_to - (stock + on_order), 0.0)
        if qty > 0:
            pipeline[day + max(int(round(lead_time)), 1)] = pipeline.get(day + max(int(round(lead_time)), 1), 0.0) + qty
            orders += 1
            units_ordered += qty
        holding += float(daily_holding_cost(stock, unit_price))
    return {"shortage": shortage, "holding": holding, "orders": orders,
            "units_ordered": units_ordered, "ending_stock": stock}


def simulate():
    retail = pd.read_csv(RETAIL_PATH, parse_dates=["date"])
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    inventory = pd.read_csv(INVENTORY_PATH)
    start_date = forecast["date"].min()

    latest = (retail.sort_values("date").groupby(["store", "product"], as_index=False).tail(1)
              [["store", "product", "reorder_point", "reorder_qty", "lead_time_days"]]
              .rename(columns={"reorder_point": "hist_reorder_point", "reorder_qty": "hist_reorder_qty",
                               "lead_time_days": "planned_lead_time"}))
    open_orders = order_history(retail).query("open")
    open_orders = open_orders.assign(
        arrival_day=((open_orders["order_date"] + pd.to_timedelta(open_orders["planned_lead_time"], unit="D")) - start_date).dt.days.clip(lower=0))

    params = inventory.merge(latest, on=["store", "product"], how="left")
    rows = []
    for r in params.itertuples(index=False):
        series = forecast[(forecast["store"] == r.store) & (forecast["product"] == r.product)].sort_values("date")
        if series.empty:
            continue
        demand = series.forecast_demand.to_numpy(dtype=float)
        price = float(series.unit_price.mean())
        transit = open_orders[(open_orders["store"] == r.store) & (open_orders["product"] == r.product)]
        arrivals = dict(zip(transit["arrival_day"].astype(int), transit["ordered_qty"]))
        start = max(float(r.current_stock), 0.0)

        none = simulate_policy(demand, start, arrivals, r.planned_lead_time, price, "none")
        hist = simulate_policy(demand, start, arrivals, r.planned_lead_time, price, "fixed_qty_on_hand",
                               reorder_point=r.hist_reorder_point, fixed_qty=r.hist_reorder_qty)
        rec = simulate_policy(demand, start, arrivals, r.planned_lead_time, price, "order_up_to_ip",
                              reorder_point=r.reorder_point, order_up_to=r.order_up_to_level)

        reduction = hist["shortage"] - rec["shortage"]
        avoided_revenue = reduction * price
        avoided_gross_profit = avoided_revenue * (1 - UNIT_COST_RATE)
        incremental_holding = rec["holding"] - hist["holding"]
        incremental_ordering = (rec["orders"] - hist["orders"]) * ORDERING_COST
        rows.append({
            "store": r.store, "product": r.product,
            "no_order_stockout_units": round(none["shortage"], 2),
            "baseline_stockout_units": round(hist["shortage"], 2),
            "recommended_policy_stockout_units": round(rec["shortage"], 2),
            "stockout_reduction_units": round(reduction, 2),
            "recommended_order_qty": round(float(r.recommended_order_qty), 2),
            "baseline_orders": hist["orders"], "recommended_orders": rec["orders"],
            "baseline_units_ordered": round(hist["units_ordered"], 2),
            "recommended_units_ordered": round(rec["units_ordered"], 2),
            "avoided_lost_sales_value": round(avoided_revenue, 2),
            "avoided_lost_gross_profit": round(avoided_gross_profit, 2),
            "baseline_holding_cost": round(hist["holding"], 2),
            "recommended_holding_cost": round(rec["holding"], 2),
            "incremental_holding_cost": round(incremental_holding, 2),
            "incremental_ordering_cost": round(incremental_ordering, 2),
            "estimated_net_benefit": round(avoided_gross_profit - incremental_holding - incremental_ordering, 2),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No forecast/inventory pairs were available for simulation.")
    result.to_csv(OUTPUT_PATH, index=False)

    cols = ["no_order_stockout_units", "baseline_stockout_units", "recommended_policy_stockout_units",
            "stockout_reduction_units", "baseline_orders", "recommended_orders",
            "avoided_lost_sales_value", "avoided_lost_gross_profit", "baseline_holding_cost",
            "recommended_holding_cost", "incremental_holding_cost", "incremental_ordering_cost",
            "estimated_net_benefit"]
    summary = result[cols].sum()
    print("30-day simulation: recommended policy vs historical reorder rule")
    print(summary.round(2).to_string())
    base = summary["baseline_stockout_units"]
    if base:
        print(f"Stockout reduction vs historical rule: {100 * summary['stockout_reduction_units'] / base:.1f}%")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    simulate()
