"""Walk-forward backtesting for SKU/store demand forecasting.

Two evaluations from the same three chronological origins:

1. One-step-ahead (forecast_backtest_results.csv): every prediction uses the
   real demand of the previous day, like the holdout metrics.
2. Recursive 30-day (forecast_horizon_backtest.csv): the exact procedure that
   produces the planning forecast used by inventory. Day h is predicted from
   the model's own earlier predictions, so errors compound. Reported by
   horizon against a seasonal-naive forecast that repeats the last observed week.
   Promotions are treated as known in advance (they are planned), so the actual
   promotion calendar is passed as the scenario.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from common import FEATURES, evaluate, feature_table, load_daily, make_model, recursive_forecast

DATA_PATH = "data/retail_sales_data.csv"
OUTPUT_PATH = Path("data/forecast_backtest_results.csv")
HORIZON_PATH = Path("data/forecast_horizon_backtest.csv")
HORIZON = 30

# Three sequential 30-day evaluation windows. Each fold trains only on dates
# strictly before the validation window, preventing future leakage.
FOLDS = [30, 60, 90]


def seasonal_naive_recursive(daily, start_date, horizon):
    """Repeat each series' last observed 7 days across the horizon."""
    hist = daily[daily.date < start_date].sort_values("date")
    rows = []
    for (store, product), g in hist.groupby(["store", "product"], sort=False):
        last_week = g.demand.to_numpy(dtype=float)[-7:]
        for h in range(horizon):
            rows.append({"date": start_date + pd.Timedelta(days=h), "store": store, "product": product,
                         "naive_forecast": last_week[h % 7]})
    return pd.DataFrame(rows)


def main():
    daily = load_daily(DATA_PATH)
    data = feature_table(daily)
    max_date = data.date.max()
    rows, horizon_rows = [], []

    for days_back in FOLDS:
        test_end = max_date - pd.DateOffset(days=days_back)
        test_start = test_end - pd.DateOffset(days=HORIZON - 1)
        train = data[data.date < test_start]
        test = data[(data.date >= test_start) & (data.date <= test_end)]
        if train.empty or test.empty:
            continue
        fold = f"T-{days_back + HORIZON - 1}:T-{days_back}"
        meta = {"fold": fold, "train_end": train.date.max().date(), "test_start": test.date.min().date(),
                "test_end": test.date.max().date(), "rows": len(test)}

        fitted = make_model()
        fitted.fit(train[FEATURES], train.demand, verbose=False)

        # 1) one-step-ahead
        prediction = np.maximum(fitted.predict(test[FEATURES]), 0)
        rows.append({**evaluate("Seasonal-Naive-7-Day", test.demand, test["lag_7"]), **meta})
        rows.append({**evaluate("SKU-Store XGBoost", test.demand, prediction), **meta})

        # 2) recursive 30-day, the procedure used for planning
        promo = daily[(daily.date >= test_start) & (daily.date <= test_end)]
        lookup = promo.set_index(["date", "store", "product"])[["promo_event", "discount_pct"]].to_dict("index")
        rec = recursive_forecast(fitted, daily, test_start, HORIZON, lookup)
        rec = rec.merge(seasonal_naive_recursive(daily, test_start, HORIZON), on=["date", "store", "product"])
        rec = rec.merge(daily[["date", "store", "product", "demand"]], on=["date", "store", "product"])
        rows.append({**evaluate("SKU-Store XGBoost (recursive 30-day)", rec.demand, rec.forecast_demand), **meta})
        rows.append({**evaluate("Seasonal-Naive (recursive 30-day)", rec.demand, rec.naive_forecast), **meta})

        for h, part in rec.groupby("horizon"):
            horizon_rows.append({"fold": fold, "horizon": h,
                                 "xgb_MAE": float(np.mean(np.abs(part.demand - part.forecast_demand))),
                                 "naive_MAE": float(np.mean(np.abs(part.demand - part.naive_forecast))),
                                 "xgb_bias": float(np.mean(part.forecast_demand - part.demand)),
                                 "actual_mean": float(part.demand.mean())})

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No valid walk-forward folds were produced.")
    result.to_csv(OUTPUT_PATH, index=False)

    horizon = pd.DataFrame(horizon_rows)
    horizon.to_csv(HORIZON_PATH, index=False)

    print("Walk-forward average metrics:")
    print(result.groupby("model")[["MAE", "RMSE", "MAPE", "WAPE"]].mean().round(3).to_string())
    by_h = horizon.groupby("horizon")[["xgb_MAE", "naive_MAE", "xgb_bias"]].mean()
    print("\nRecursive forecast error by horizon (mean of folds):")
    print(by_h.loc[[1, 7, 14, 30]].round(3).to_string())
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {HORIZON_PATH}")


if __name__ == "__main__":
    main()
