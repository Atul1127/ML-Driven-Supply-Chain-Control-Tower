"""Shared assumptions, features, metrics and helpers for the pipeline.

Keeping these in one place means the forecasting, backtesting, inventory and
simulation steps cannot silently disagree on a feature definition or a cost
assumption (the old simulation charged a yearly holding rate every day).
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Business assumptions (single source of truth)
# ---------------------------------------------------------------------------
UNIT_COST_RATE = 0.60          # unit cost = 60% of list price (no procurement cost in data)
ANNUAL_HOLDING_RATE = 0.25     # holding cost = 25% of inventory cost value per YEAR
DAILY_HOLDING_RATE = ANNUAL_HOLDING_RATE / 365.0
ORDERING_COST = 2000.0         # fixed cost per purchase order (assumption)
SERVICE_LEVEL_Z = 1.645        # 95% cycle-service level

FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_30", "rolling_mean_7", "rolling_mean_30",
    "rolling_std_7", "day_of_week", "month", "day_of_month", "is_weekend",
    "promo_event", "discount_pct",
]
PROMO_FEATURES = ["promo_event", "discount_pct"]


# ---------------------------------------------------------------------------
# Cost formulas
# ---------------------------------------------------------------------------
def unit_cost(unit_price):
    return np.asarray(unit_price, dtype=float) * UNIT_COST_RATE


def daily_holding_cost(stock_units, unit_price):
    """Holding cost for ONE day of stock: units x unit cost x (25% / 365)."""
    return np.asarray(stock_units, dtype=float) * unit_cost(unit_price) * DAILY_HOLDING_RATE


def annual_holding_cost_per_unit(unit_price):
    return unit_cost(unit_price) * ANNUAL_HOLDING_RATE


def eoq(annual_demand, unit_price, ordering_cost=ORDERING_COST):
    """Economic order quantity with holding cost on unit COST, not selling price."""
    h = np.clip(annual_holding_cost_per_unit(unit_price), 0.01, None)
    d = np.clip(np.asarray(annual_demand, dtype=float), 1, None)
    return np.sqrt(2 * d * ordering_cost / h)


def safety_stock(sigma_daily_error, mean_lead_time, daily_demand, sigma_lead_time, z=SERVICE_LEVEL_Z):
    """z * sqrt(L * sigma_e^2 + d^2 * sigma_L^2).

    sigma_daily_error is the spread of daily FORECAST ERROR (not raw demand), so
    trend/seasonality/promotions the model already predicts are not buffered twice.
    """
    L = np.clip(np.asarray(mean_lead_time, dtype=float), 1, None)
    s_e = np.asarray(sigma_daily_error, dtype=float)
    d = np.asarray(daily_demand, dtype=float)
    s_l = np.asarray(sigma_lead_time, dtype=float)
    return z * np.sqrt(L * s_e ** 2 + d ** 2 * s_l ** 2)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def mape(actual, prediction):
    actual, prediction = np.asarray(actual, float), np.asarray(prediction, float)
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - prediction[mask]) / actual[mask])) * 100) if mask.any() else 0.0


def wape(actual, prediction):
    """Total absolute error / total demand. Robust to zero-demand days."""
    actual, prediction = np.asarray(actual, float), np.asarray(prediction, float)
    denom = np.abs(actual).sum()
    return float(np.abs(actual - prediction).sum() / denom * 100) if denom else 0.0


def evaluate(name, actual, prediction, **extra):
    actual, prediction = np.asarray(actual, float), np.asarray(prediction, float)
    return {
        "model": name,
        "MAE": float(np.mean(np.abs(actual - prediction))),
        "RMSE": float(np.sqrt(np.mean((actual - prediction) ** 2))),
        "MAPE": mape(actual, prediction),
        "WAPE": wape(actual, prediction),
        **extra,
    }


# ---------------------------------------------------------------------------
# Data + features
# ---------------------------------------------------------------------------
def load_daily(path="data/retail_sales_data.csv"):
    raw = pd.read_csv(path, parse_dates=["date"])
    required = {"store", "product", "category", "date", "demand", "promo_event", "discount_pct", "unit_price"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return raw.groupby(["store", "product", "category", "date"], as_index=False).agg(
        demand=("demand", "sum"), promo_event=("promo_event", "max"),
        discount_pct=("discount_pct", "mean"), unit_price=("unit_price", "mean"),
    )


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


def feature_table(daily):
    parts = [build_features(g) for _, g in daily.groupby(["store", "product"], sort=False) if len(g) > 60]
    if not parts:
        raise ValueError("No store/product series has enough history for lag features.")
    return pd.concat(parts, ignore_index=True).dropna(subset=FEATURES)


def make_model():
    return XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=-1,
    )


def recursive_forecast(model, daily, start_date, horizon=30, promo_lookup=None, features=FEATURES):
    """Recursive multi-day forecast for every store/product series at once.

    Day h uses the predictions for days < h as its lags (exactly how the 30-day
    planning forecast is produced). Only history strictly before start_date is
    used. promo_lookup maps (date, store, product) -> {promo_event, discount_pct};
    missing days default to no promotion.
    """
    promo_lookup = promo_lookup or {}
    start_date = pd.Timestamp(start_date)
    hist = daily[daily.date < start_date].sort_values("date")
    keys, arrays = [], []
    for (store, product), g in hist.groupby(["store", "product"], sort=False):
        if len(g) < 30:
            continue
        keys.append((store, product, g.category.iloc[-1], float(g.unit_price.iloc[-1])))
        arrays.append(list(g.demand.to_numpy(dtype=float)[-30:]))

    out = []
    for h in range(horizon):
        date = start_date + pd.Timedelta(days=h)
        rows = []
        for (store, product, category, price), arr in zip(keys, arrays):
            a = np.asarray(arr[-30:])
            planned = promo_lookup.get((date, store, product), {"promo_event": 0, "discount_pct": 0.0})
            rows.append({
                "date": date, "store": store, "product": product, "category": category, "unit_price": price,
                "horizon": h + 1,
                "promo_event": int(planned["promo_event"]), "discount_pct": float(planned["discount_pct"]),
                "lag_1": a[-1], "lag_7": a[-7], "lag_14": a[-14], "lag_30": a[-30],
                "rolling_mean_7": a[-7:].mean(), "rolling_mean_30": a.mean(),
                "rolling_std_7": a[-7:].std(ddof=1), "day_of_week": date.dayofweek,
                "month": date.month, "day_of_month": date.day, "is_weekend": int(date.dayofweek >= 5),
            })
        frame = pd.DataFrame(rows)
        pred = np.maximum(model.predict(frame[features]), 0)
        for arr, value in zip(arrays, pred):
            arr.append(float(value))
        frame["forecast_demand"] = pred
        out.append(frame)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Purchase orders: match each order to its delivery
# ---------------------------------------------------------------------------
def order_history(df):
    """One row per purchase order, matched to its receipt.

    The dataset records an order on the order day and the delivery on the
    arrival day (different rows). Each store/product has at most one order in
    transit, so the k-th order pairs with the k-th receipt. The last order
    with no receipt yet is still in transit (on order).
    """
    rows = []
    for (store, product), g in df.sort_values("date").groupby(["store", "product"], sort=False):
        orders = g[g.ordered_qty > 0]
        receipts = g[g.received_qty > 0]
        for k, o in enumerate(orders.itertuples(index=False)):
            r = receipts.iloc[k] if k < len(receipts) else None
            realized = (r.date - o.date).days if r is not None else np.nan
            rows.append({
                "store": store, "product": product, "supplier": o.supplier,
                "order_date": o.date, "receipt_date": r.date if r is not None else pd.NaT,
                "ordered_qty": float(o.ordered_qty),
                "received_qty": float(r.received_qty) if r is not None else np.nan,
                "planned_lead_time": float(o.lead_time_days),
                "realized_lead_time": realized,
                "unit_price": float(o.unit_price),
            })
    orders = pd.DataFrame(rows)
    orders["open"] = orders["receipt_date"].isna()
    orders["on_time"] = np.where(orders["open"], np.nan, (orders["realized_lead_time"] <= orders["planned_lead_time"]).astype(float))
    orders["fill_rate"] = orders["received_qty"] / orders["ordered_qty"]
    return orders
