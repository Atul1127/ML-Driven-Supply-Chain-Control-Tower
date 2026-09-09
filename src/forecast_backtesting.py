"""Walk-forward backtesting for SKU/store demand forecasting.

Evaluates the same XGBoost feature set used by the production forecast across
multiple chronological windows instead of relying on one lucky holdout period.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

DATA_PATH = Path("data/retail_sales_data.csv")
OUTPUT_PATH = Path("data/forecast_backtest_results.csv")

FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_30", "rolling_mean_7", "rolling_mean_30",
    "rolling_std_7", "day_of_week", "month", "day_of_month", "is_weekend",
    "promo_event", "discount_pct",
]


# Three sequential 30-day evaluation windows. Each fold trains only on dates
# strictly before the validation window, preventing future leakage.
FOLDS = [30, 60, 90]


def mape(actual, prediction):
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - prediction[mask]) / actual[mask])) * 100) if mask.any() else 0.0


def build_features(group):
    x = group.sort_values("date").copy()
    x["lag_1"] = x.demand.shift(1)
    x["lag_7"] = x.demand.shift(7)
    x["lag_14"] = x.demand.shift(14)
    x["lag_30"] = x.demand.shift(30)
    shifted = x.demand.shift(1)
    x["rolling_mean_7"] = shifted.rolling(7).mean()
    x["rolling_mean_30"] = shifted.rolling(30).mean()
    x["rolling_std_7"] = shifted.rolling(7).std()
    x["day_of_week"] = x.date.dt.dayofweek
    x["month"] = x.date.dt.month
    x["day_of_month"] = x.date.dt.day
    x["is_weekend"] = (x.day_of_week >= 5).astype(int)
    return x


def model():
    return XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, n_jobs=-1,
    )


def main():
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    daily = raw.groupby(
        ["store", "product", "category", "date"], as_index=False
    ).agg(
        demand=("demand", "sum"),
        promo_event=("promo_event", "sum"),
        discount_pct=("discount_pct", "mean"),
        unit_price=("unit_price", "mean"),
    )

    parts = [build_features(g) for _, g in daily.groupby(["store", "product"], sort=False) if len(g) > 60]
    data = pd.concat(parts, ignore_index=True).dropna(subset=FEATURES)
    max_date = data.date.max()
    rows = []

    for days_back in FOLDS:
        test_end = max_date - pd.DateOffset(days=days_back)
        test_start = test_end - pd.DateOffset(days=29)
        train = data[data.date < test_start]
        test = data[(data.date >= test_start) & (data.date <= test_end)]
        if train.empty or test.empty:
            continue

        fitted = model()
        fitted.fit(train[FEATURES], train.demand, verbose=False)
        prediction = np.maximum(fitted.predict(test[FEATURES]), 0)

        seasonal = test["lag_7"]
        rows.extend([
            {
                "fold": f"T-{days_back + 29}:T-{days_back}",
                "model": "Seasonal-Naive-7-Day",
                "train_end": train.date.max().date(),
                "test_start": test.date.min().date(),
                "test_end": test.date.max().date(),
                "rows": len(test),
                "MAE": mean_absolute_error(test.demand, seasonal),
                "RMSE": np.sqrt(mean_squared_error(test.demand, seasonal)),
                "MAPE": mape(test.demand, seasonal),
            },
            {
                "fold": f"T-{days_back + 29}:T-{days_back}",
                "model": "SKU-Store XGBoost",
                "train_end": train.date.max().date(),
                "test_start": test.date.min().date(),
                "test_end": test.date.max().date(),
                "rows": len(test),
                "MAE": mean_absolute_error(test.demand, prediction),
                "RMSE": np.sqrt(mean_squared_error(test.demand, prediction)),
                "MAPE": mape(test.demand, prediction),
            },
        ])

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No valid walk-forward folds were produced.")
    result.to_csv(OUTPUT_PATH, index=False)

    summary = result.groupby("model")[["MAE", "RMSE", "MAPE"]].mean().round(3)
    print("Walk-forward average metrics:")
    print(summary.to_string())
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
