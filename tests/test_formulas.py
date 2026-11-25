"""Unit tests for the decision-layer formulas (run: python -m pytest -q)."""

import numpy as np
import pandas as pd
import pytest

from business_impact_simulation import simulate_policy
from common import (ANNUAL_HOLDING_RATE, SERVICE_LEVEL_Z, UNIT_COST_RATE, daily_holding_cost, eoq,
                    order_history, safety_stock, wape)
from finance_analysis import finance_summary


# --- holding cost -----------------------------------------------------------
def test_daily_holding_cost_is_annual_rate_over_365():
    # 1,000 units at price 100 -> cost value 60,000 -> 25%/yr -> 15,000/yr
    assert daily_holding_cost(1000, 100) == pytest.approx(15000 / 365)


def test_a_year_of_daily_holding_equals_annual_rate():
    one_year = sum(daily_holding_cost(500, 40) for _ in range(365))
    assert one_year == pytest.approx(500 * 40 * UNIT_COST_RATE * ANNUAL_HOLDING_RATE)


def test_simulated_holding_matches_formula():
    out = simulate_policy(np.zeros(30), start_stock=100, arrivals={}, lead_time=5, unit_price=10, rule="none")
    assert out["holding"] == pytest.approx(30 * 100 * 10 * 0.60 * 0.25 / 365)


# --- inventory policy -------------------------------------------------------
def test_safety_stock_without_lead_time_variance_is_textbook():
    assert safety_stock(5.0, 9.0, 30.0, 0.0) == pytest.approx(SERVICE_LEVEL_Z * 5.0 * 3.0)


def test_lead_time_variance_increases_safety_stock():
    assert safety_stock(5.0, 9.0, 30.0, 0.5) > safety_stock(5.0, 9.0, 30.0, 0.0)


def test_eoq_uses_unit_cost_not_price():
    expected = np.sqrt(2 * 3650 * 2000 / (100 * UNIT_COST_RATE * ANNUAL_HOLDING_RATE))
    assert eoq(3650, 100) == pytest.approx(expected)


def test_order_up_to_counts_in_transit_stock():
    # stock 0, 80 units arriving day 2, ROP 100, S 150: order must be 150 - 80 = 70, not 150
    out = simulate_policy(np.zeros(1), start_stock=0, arrivals={2: 80}, lead_time=5, unit_price=10,
                          rule="order_up_to_ip", reorder_point=100, order_up_to=150)
    assert out["units_ordered"] == pytest.approx(70)


def test_no_order_when_position_above_reorder_point():
    out = simulate_policy(np.zeros(1), start_stock=10, arrivals={1: 100}, lead_time=5, unit_price=10,
                          rule="order_up_to_ip", reorder_point=50, order_up_to=150)
    assert out["orders"] == 0


def test_historical_rule_allows_one_order_in_transit():
    out = simulate_policy(np.full(5, 10.0), start_stock=0, arrivals={}, lead_time=10, unit_price=10,
                          rule="fixed_qty_on_hand", reorder_point=20, fixed_qty=100)
    assert out["orders"] == 1


# --- data plumbing ----------------------------------------------------------
def test_order_history_matches_orders_to_receipts():
    dates = pd.date_range("2024-01-01", periods=12)
    df = pd.DataFrame({"date": dates, "store": "S", "product": "P", "supplier": "X", "unit_price": 10,
                       "lead_time_days": 3, "ordered_qty": 0, "received_qty": 0})
    df.loc[0, "ordered_qty"] = 50; df.loc[3, "received_qty"] = 50   # on time
    df.loc[4, "ordered_qty"] = 40; df.loc[9, "received_qty"] = 40   # 2 days late
    df.loc[10, "ordered_qty"] = 30                                  # still in transit
    o = order_history(df)
    assert list(o["realized_lead_time"].iloc[:2]) == [3, 5]
    assert list(o["on_time"].iloc[:2]) == [1.0, 0.0]
    assert o["open"].tolist() == [False, False, True]


def test_finance_dio_uses_total_inventory_per_day():
    days = pd.date_range("2024-01-01", periods=365)
    rows = []
    for store in ["A", "B"]:
        rows.append(pd.DataFrame({"date": days, "store": store, "revenue": 100.0, "cogs": 60.0,
                                  "inventory_value": 600.0, "lost_sales_value": 0.0, "lost_gross_profit": 0.0}))
    df = pd.concat(rows)
    out = finance_summary(df).iloc[0]
    # annual COGS 43,800 / avg total inventory 1,200 -> turnover 36.5 -> DIO 10 days
    assert out["inventory_turnover"] == pytest.approx(36.5)
    assert out["days_inventory_outstanding"] == pytest.approx(10.0)


def test_wape_handles_zero_demand_days():
    # (|0-1| + |10-9|) / (0 + 10) = 20%; MAPE would have to drop the zero day
    assert wape([0, 10], [1, 9]) == pytest.approx(20.0)
