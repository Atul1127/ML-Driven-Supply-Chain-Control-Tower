"""Generate SHAP feature importance for the SKU/store XGBoost forecaster."""

import pandas as pd
import shap

from common import FEATURES, feature_table, load_daily, make_model

DATA_PATH = "data/retail_sales_data.csv"
OUTPUT_PATH = "data/sku_xgboost_shap_importance.csv"


def main():
    data = feature_table(load_daily(DATA_PATH))
    cutoff = data.date.max() - pd.Timedelta(days=60)
    train = data[data.date <= cutoff]

    model = make_model()
    model.fit(train[FEATURES], train.demand, verbose=False)
    sample = train[FEATURES].tail(min(5000, len(train)))
    values = shap.TreeExplainer(model).shap_values(sample)
    importance = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(OUTPUT_PATH, index=False)
    print(importance.to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
