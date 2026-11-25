"""SKU/store-level naive baselines for fair forecasting benchmarks.

Includes a promotion-adjusted seasonal naive: XGBoost knows whether today is a
promotion, so a fair baseline should too. The promotion uplift is estimated on
the training period only.
"""

import numpy as np
import pandas as pd

from common import evaluate

DATA_PATH = "data/retail_sales_data.csv"
OUTPUT_PATH = "data/baseline_results.csv"
HOLDOUT_DAYS = 60


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    daily = df.groupby(["store", "product", "date"], as_index=False).agg(
        demand=("demand", "sum"), promo_event=("promo_event", "max"))
    daily = daily.sort_values(["store", "product", "date"])
    g = daily.groupby(["store", "product"])
    daily["lag_1"] = g.demand.shift(1)
    daily["lag_7"] = g.demand.shift(7)
    daily["promo_lag_7"] = g.promo_event.shift(7)
    daily = daily.dropna()

    cutoff = daily.date.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train, test = daily[daily.date <= cutoff], daily[daily.date > cutoff].copy()
    if test.empty:
        raise ValueError("Temporal split produced an empty test set.")

    # Multiplicative promotion uplift learned on training data only.
    uplift = train.loc[train.promo_event == 1, "demand"].mean() / train.loc[train.promo_event == 0, "demand"].mean()
    factor_today = np.where(test.promo_event == 1, uplift, 1.0)
    factor_lag = np.where(test.promo_lag_7 == 1, uplift, 1.0)
    test["promo_adjusted_lag_7"] = test.lag_7 / factor_lag * factor_today

    results = pd.DataFrame([
        evaluate("SKU Naive-1-Day", test.demand, test.lag_1),
        evaluate("SKU Seasonal-Naive-7-Day", test.demand, test.lag_7),
        evaluate("SKU Seasonal-Naive-7-Day (promo-adjusted)", test.demand, test.promo_adjusted_lag_7),
    ])
    results["sku_store_pairs"] = daily[["store", "product"]].drop_duplicates().shape[0]
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"Training-period promotion uplift: {uplift:.3f}x")
    print(results.round(4).to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
