"""Temporal supplier/SKU deterioration signals.

Fixes vs the earlier version:
- Lead time, on-time and fill rate come from REALIZED purchase orders (each order
  matched to its delivery), credited on the delivery date. The old daily fill
  rate divided a day's receipts by the same day's orders, which almost never
  fall on the same row, so it was ~always 0. The old lead time / on-time /
  defect inputs were the generator's constant settings and never moved.
- Recent performance uses a 28-day window of deliveries; the baseline is the
  90 days before that window. A metric only contributes when the recent window
  has at least MIN_ORDERS deliveries.
- Defect rate is excluded from the signal: the dataset only holds a constant
  configured rate per supplier, so it carries no temporal information.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from common import order_history

DATA_PATH = Path("data/retail_sales_data.csv")
OUTPUT_PATH = Path("data/temporal_disruption.csv")
RECENT_DAYS = 28
BASELINE_DAYS = 90
MIN_ORDERS = 2
WEIGHTS = {"lead_time": 40, "on_time": 30, "fill_rate": 15, "stockout": 15}


def run():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    required = {"date", "store", "supplier", "product", "lead_time_days", "ordered_qty", "received_qty", "demand", "stockout"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keys = ["supplier", "product"]
    daily = df.groupby(keys + ["date"], as_index=False).agg(demand=("demand", "sum"), stockout=("stockout", "mean"))

    orders = order_history(df)
    delivered = orders[~orders["open"]].copy()
    delivered["late"] = 1 - delivered["on_time"]
    events = delivered.groupby(keys + ["receipt_date"], as_index=False).agg(
        n_orders=("ordered_qty", "size"), lead_sum=("realized_lead_time", "sum"), late_sum=("late", "sum"),
        received_sum=("received_qty", "sum"), ordered_sum=("ordered_qty", "sum"),
    ).rename(columns={"receipt_date": "date"})
    daily = daily.merge(events, on=keys + ["date"], how="left")
    sums = ["n_orders", "lead_sum", "late_sum", "received_sum", "ordered_sum"]
    daily[sums] = daily[sums].fillna(0)
    daily = daily.sort_values(keys + ["date"]).reset_index(drop=True)

    g = daily.groupby(keys, group_keys=False)
    for c in sums + ["stockout"]:
        daily[f"{c}_recent"] = g[c].transform(lambda s: s.rolling(RECENT_DAYS, min_periods=1).sum())
        daily[f"{c}_base"] = g[c].transform(lambda s: s.shift(RECENT_DAYS).rolling(BASELINE_DAYS, min_periods=30).sum())
    daily["stockout_recent"] /= RECENT_DAYS
    daily["stockout_base"] /= BASELINE_DAYS

    def ratio(num, den):
        return daily[num] / daily[den].replace(0, np.nan)

    daily["lead_time"] = ratio("lead_sum_recent", "n_orders_recent")
    daily["lead_time_baseline"] = ratio("lead_sum_base", "n_orders_base")
    daily["on_time"] = 1 - ratio("late_sum_recent", "n_orders_recent")
    daily["on_time_baseline"] = 1 - ratio("late_sum_base", "n_orders_base")
    daily["fill_rate"] = ratio("received_sum_recent", "ordered_sum_recent")
    daily["fill_rate_baseline"] = ratio("received_sum_base", "ordered_sum_base")
    daily["stockout_baseline"] = daily["stockout_base"]
    daily["stockout"] = daily["stockout_recent"]

    enough = daily["n_orders_recent"] >= MIN_ORDERS
    worse = pd.DataFrame(index=daily.index)
    worse["lead_time"] = ((daily["lead_time"] - daily["lead_time_baseline"]) / daily["lead_time_baseline"]).where(enough)
    worse["on_time"] = (daily["on_time_baseline"] - daily["on_time"]).where(enough)            # drop in on-time share
    worse["fill_rate"] = (daily["fill_rate_baseline"] - daily["fill_rate"]).where(enough)      # drop in fill share
    worse["stockout"] = daily["stockout"] - daily["stockout_baseline"]                         # rise in stockout share
    worse = worse.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)

    # Each component is scaled so a plausible "bad" move maps to its full weight:
    # +10% lead time, -30pp on-time, -20pp fill, +30pp stockout days.
    scale = {"lead_time": 0.10, "on_time": 0.30, "fill_rate": 0.20, "stockout": 0.30}
    signal = sum(WEIGHTS[k] * (worse[k] / scale[k]).clip(upper=1) for k in WEIGHTS)
    for k in WEIGHTS:
        daily[f"{k}_deterioration"] = worse[k].round(4)
    daily["disruption_signal"] = signal.clip(0, 100).round(2)
    daily.loc[daily["lead_time_baseline"].isna() & daily["stockout_baseline"].isna(), "disruption_signal"] = 0.0
    daily["disruption_stage"] = pd.cut(daily["disruption_signal"], bins=[-np.inf, 20, 40, 70, np.inf],
                                       labels=["NORMAL", "EARLY_WARNING", "EMERGING", "CRITICAL"], right=False)
    daily["is_disruption"] = daily["disruption_stage"].isin(["EMERGING", "CRITICAL"])

    out_cols = keys + ["date", "demand", "n_orders_recent", "lead_time", "lead_time_baseline", "on_time", "on_time_baseline",
                       "fill_rate", "fill_rate_baseline", "stockout", "stockout_baseline"] + \
               [f"{k}_deterioration" for k in WEIGHTS] + ["disruption_signal", "disruption_stage", "is_disruption"]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily[out_cols].to_csv(OUTPUT_PATH, index=False)
    share = daily["disruption_stage"].value_counts(normalize=True).mul(100).round(2)
    print("Share of supplier-product-days by stage (%):")
    print(share.to_string())
    print(f"Saved temporal disruption signals: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
