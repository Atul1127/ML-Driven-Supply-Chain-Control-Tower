"""SKU/store-level demand forecasting with leakage-safe lag features.

Outputs
- sku_xgboost_results.csv: one-day-ahead holdout metrics for the main model and
  a no-promotion-feature ablation (so the gain over naive baselines can be split
  into "model" vs "promotion information").
- sku_test_predictions.csv: holdout predictions; residuals drive safety stock.
- sku_30_day_forecast.csv: recursive 30-day planning forecast.
"""

import os
import numpy as np
import pandas as pd

from common import (FEATURES, PROMO_FEATURES, evaluate, feature_table, load_daily,
                    make_model, recursive_forecast)

DATA_PATH = "data/retail_sales_data.csv"
SCENARIO_PATH = "data/forecast_scenario.csv"
RESULTS_PATH = "data/sku_xgboost_results.csv"
TEST_PRED_PATH = "data/sku_test_predictions.csv"
FORECAST_PATH = "data/sku_30_day_forecast.csv"
OUTPUT_DIR = "images/forecasting"
HOLDOUT_DAYS = 60
HORIZON = 30


def load_scenario(start_date, pairs):
    """Load optional known future promotions/discounts; unspecified days stay at zero."""
    if not os.path.exists(SCENARIO_PATH):
        return pd.DataFrame(columns=["date", "store", "product", "promo_event", "discount_pct"])
    scenario = pd.read_csv(SCENARIO_PATH, comment="#", parse_dates=["date"])
    required = {"date", "store", "product", "promo_event", "discount_pct"}
    if not required.issubset(scenario.columns):
        raise ValueError(f"{SCENARIO_PATH} must contain {sorted(required)}")
    scenario = scenario[scenario.date >= start_date].copy()
    scenario = scenario.merge(pairs, on=["store", "product"], how="inner")
    scenario["promo_event"] = scenario["promo_event"].clip(0, 1).astype(int)
    scenario["discount_pct"] = scenario["discount_pct"].clip(0, 100)
    return scenario[["date", "store", "product", "promo_event", "discount_pct"]]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    daily = load_daily(DATA_PATH)
    data = feature_table(daily)

    cutoff = data.date.max() - pd.DateOffset(days=HOLDOUT_DAYS)
    train, test = data[data.date <= cutoff], data[data.date > cutoff]
    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty train or test set.")
    pairs_count = data[["store", "product"]].drop_duplicates().shape[0]

    model = make_model()
    model.fit(train[FEATURES], train.demand, verbose=False)
    pred = np.maximum(model.predict(test[FEATURES]), 0)

    # Ablation: same model without today's promotion/discount. The naive
    # baselines never see promotions, so this is the like-for-like comparison.
    no_promo_features = [f for f in FEATURES if f not in PROMO_FEATURES]
    ablation = make_model()
    ablation.fit(train[no_promo_features], train.demand, verbose=False)
    pred_no_promo = np.maximum(ablation.predict(test[no_promo_features]), 0)

    extra = {"train_rows": len(train), "test_rows": len(test), "sku_store_pairs": pairs_count}
    results = pd.DataFrame([
        evaluate("SKU-Store XGBoost", test.demand, pred, **extra),
        evaluate("SKU-Store XGBoost (no promo features)", test.demand, pred_no_promo, **extra),
    ])
    results.to_csv(RESULTS_PATH, index=False)

    test_out = test[["date", "store", "product", "category", "demand"]].copy()
    test_out["predicted_demand"] = pred
    test_out.to_csv(TEST_PRED_PATH, index=False)

    # Retrain on all history for the planning forecast.
    final = make_model()
    final.fit(data[FEATURES], data.demand, verbose=False)
    pairs = daily[["store", "product"]].drop_duplicates()
    start_date = daily.date.max() + pd.DateOffset(days=1)
    scenario = load_scenario(start_date, pairs)
    lookup = scenario.set_index(["date", "store", "product"])[["promo_event", "discount_pct"]].to_dict("index")

    forecast = recursive_forecast(final, daily, start_date, HORIZON, lookup)
    if forecast.empty:
        raise ValueError("No SKU/store forecasts were generated.")
    forecast["forecast_demand"] = forecast["forecast_demand"].round(2)
    forecast[["date", "store", "product", "category", "forecast_demand", "unit_price", "promo_event", "discount_pct"]].to_csv(FORECAST_PATH, index=False)

    print(results.round(4).to_string(index=False))
    if scenario.empty:
        print("Note: forecast_scenario.csv has no future promotions, so the 30-day forecast assumes none.")
    print(f"Saved: {RESULTS_PATH}")
    print(f"Saved: {FORECAST_PATH}")


if __name__ == "__main__":
    main()
