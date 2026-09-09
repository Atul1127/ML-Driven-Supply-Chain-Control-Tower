"""Executive Streamlit dashboard for the ML-driven supply-chain control tower."""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ML-Driven Supply Chain Control Tower", page_icon="📦", layout="wide")
st.title("📦 ML-Driven Supply Chain Control Tower")
st.caption("SQL analytics • Financial performance • Forecasting • Inventory optimization • Supplier risk • Decision support")


@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


control = load_csv("data/control_tower_inventory.csv")
forecast = load_csv("data/sku_30_day_forecast.csv")
risk = load_csv("data/supplier_risk_analysis.csv")
impact = load_csv("data/business_impact.csv")
baseline = load_csv("data/baseline_results.csv")
xgb_results = load_csv("data/sku_xgboost_results.csv")
backtest = load_csv("data/forecast_backtest_results.csv")
temporal = load_csv("data/temporal_disruption.csv")
finance = load_csv("data/finance_summary.csv")
budget = load_csv("data/budget_vs_actual.csv")
promo = load_csv("data/promotion_effectiveness.csv")
abc = load_csv("data/abc_analysis.csv")
supplier_finance = load_csv("data/supplier_stockout_impact.csv")
simulation = load_csv("data/replenishment_impact_simulation.csv")

if control.empty:
    st.warning("Run `python src/run_pipeline.py` first to generate the control-tower outputs.")
    st.stop()

priority = control.get("priority", pd.Series(dtype=str)).astype(str)
inventory_status = control.get("inventory_status", pd.Series(dtype=str)).astype(str)
risk_level = risk.get("risk_level", pd.Series(dtype=str)).astype(str) if not risk.empty else pd.Series(dtype=str)

kpi = st.columns(6)
kpi[0].metric("Store × SKU", f"{len(control):,}")
kpi[1].metric("High Priority", int(priority.isin(["HIGH", "URGENT"]).sum()))
kpi[2].metric("Critical Inventory", int((inventory_status == "CRITICAL").sum()))
kpi[3].metric("Reorder", int((inventory_status == "REORDER").sum()))
kpi[4].metric("High Supplier Risk", int((risk_level == "HIGH").sum()))
kpi[5].metric("Risk Signals", int(temporal.get("is_disruption", pd.Series(dtype=bool)).sum()) if not temporal.empty else 0)

st.divider()
tabs = st.tabs(["🚨 Control Tower", "💰 Financial Analytics", "📈 Sales & Revenue", "📦 Inventory", "🏭 Supplier Risk", "⚠️ Risk Signals", "🔮 Forecast", "📊 Model Benchmark", "🎯 Recommendations"])

with tabs[0]:
    st.subheader("Priority actions")
    c1, c2 = st.columns(2)
    with c1:
        store_options = ["All stores"] + sorted(control["store"].dropna().unique().tolist())
        store_filter = st.selectbox("Store", store_options)
    with c2:
        priority_filter = st.selectbox("Priority", ["All priorities", "URGENT", "HIGH", "MEDIUM", "LOW"])
    view = control.copy()
    if store_filter != "All stores":
        view = view[view["store"] == store_filter]
    if priority_filter != "All priorities":
        view = view[view["priority"] == priority_filter]
    preferred = ["priority", "action", "store", "product", "category", "supplier", "inventory_status", "days_of_stock", "shortage_to_rop", "recommended_order_qty", "average_30_day_forecast", "risk_level"]
    st.dataframe(view[[c for c in preferred if c in view.columns]], use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Financial performance")
    if finance.empty:
        st.info("Run the pipeline to generate finance outputs.")
    else:
        row = finance.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Revenue", f"₹{row['revenue']:,.0f}")
        cols[1].metric("COGS", f"₹{row['cogs']:,.0f}")
        cols[2].metric("Gross Profit", f"₹{row['gross_profit']:,.0f}")
        cols[3].metric("Gross Margin", f"{row['gross_margin_pct']:.1f}%")
        cols = st.columns(4)
        cols[0].metric("Inventory Turnover", f"{row['inventory_turnover']:.2f}x")
        cols[1].metric("DIO", f"{row['days_inventory_outstanding']:.1f} days")
        cols[2].metric("Holding Cost", f"₹{row['annual_holding_cost']:,.0f}")
        cols[3].metric("Lost-sales Exposure", f"₹{row['historical_lost_sales_value']:,.0f}")
        st.caption("Assumptions: unit cost = 60% of list price; budget values are planning assumptions.")
        if not budget.empty:
            st.markdown("### Budget vs Actual")
            st.line_chart(budget.set_index("month")[["budget_revenue", "actual_revenue"]])

with tabs[2]:
    st.subheader("Sales & revenue analytics")
    if not promo.empty:
        st.markdown("### Promotion effectiveness")
        st.dataframe(promo, use_container_width=True, hide_index=True)
        if "promo_label" in promo.columns and "average_daily_revenue" in promo.columns:
            st.bar_chart(promo.set_index("promo_label")[["average_daily_revenue"]])
    if not abc.empty:
        st.markdown("### ABC product analysis")
        st.dataframe(abc.head(30), use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Inventory analytics")
    if impact.empty:
        st.info("No inventory output found.")
    else:
        row = impact.iloc[0]
        cols = st.columns(4)
        cols[0].metric("SKU/store pairs", f"{int(row['total_sku_store_pairs']):,}")
        cols[1].metric("Current Stockouts", f"{int(row['current_stockout_pairs']):,}")
        cols[2].metric("Inventory Value", f"₹{float(row['current_inventory_value']):,.0f}")
        cols[3].metric("Recommended Units", f"{float(row.get('recommended_replenishment_units', 0)):,.0f}")
        st.caption("Historical lost-sales is simulated exposure, not savings generated by the system.")
        st.dataframe(impact, use_container_width=True, hide_index=True)
        if not simulation.empty:
            st.markdown("### 30-day replenishment policy simulation")
            totals = simulation[["baseline_stockout_units", "recommended_policy_stockout_units", "stockout_reduction_units", "avoided_lost_sales_value", "incremental_holding_cost", "estimated_net_benefit"]].sum()
            cols = st.columns(4)
            cols[0].metric("Baseline shortage", f"{totals['baseline_stockout_units']:,.0f} units")
            cols[1].metric("Policy shortage", f"{totals['recommended_policy_stockout_units']:,.0f} units")
            cols[2].metric("Avoided lost sales", f"₹{totals['avoided_lost_sales_value']:,.0f}")
            cols[3].metric("Estimated net benefit", f"₹{totals['estimated_net_benefit']:,.0f}")
            st.dataframe(simulation.sort_values("estimated_net_benefit", ascending=False).head(20), use_container_width=True, hide_index=True)
            st.caption("Planning simulation only: results depend on synthetic data and documented cost assumptions.")

with tabs[4]:
    st.subheader("Supplier performance")
    if risk.empty:
        st.info("No supplier risk output found.")
    else:
        cols = st.columns(3)
        cols[0].metric("Low", int((risk_level == "LOW").sum()))
        cols[1].metric("Medium", int((risk_level == "MEDIUM").sum()))
        cols[2].metric("High", int((risk_level == "HIGH").sum()))
        st.dataframe(risk, use_container_width=True, hide_index=True)
        if not supplier_finance.empty:
            st.markdown("### Supplier → stockout → financial exposure")
            st.dataframe(supplier_finance, use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("Temporal supplier/SKU risk signals")
    if temporal.empty:
        st.info("No temporal risk output found.")
    else:
        selected_supplier = st.selectbox("Supplier", sorted(temporal["supplier"].dropna().astype(str).unique()), key="risk_supplier")
        supplier_temporal = temporal[temporal["supplier"].astype(str) == selected_supplier].copy()
        products = sorted(supplier_temporal["product"].dropna().astype(str).unique())
        if products:
            selected_product = st.selectbox("Product", products, key="risk_product")
            timeline = supplier_temporal[supplier_temporal["product"].astype(str) == selected_product].copy()
            timeline["date"] = pd.to_datetime(timeline["date"])
            st.line_chart(timeline.sort_values("date").set_index("date")[["disruption_signal"]])
            latest = timeline.sort_values("date").iloc[-1]
            st.info(f"Latest signal: **{latest['disruption_stage']}** • Score **{latest['disruption_signal']:.1f}/100**")

with tabs[6]:
    st.subheader("30-day SKU/store demand forecast")
    if forecast.empty:
        st.info("No SKU forecast output found.")
    else:
        pairs = forecast[["store", "product"]].drop_duplicates().sort_values(["store", "product"])
        selected_pair = st.selectbox("Store × Product", [f"{r.store} | {r.product}" for r in pairs.itertuples()], key="forecast_pair")
        store, product = selected_pair.split(" | ", 1)
        view = forecast[(forecast["store"] == store) & (forecast["product"] == product)].copy()
        view["date"] = pd.to_datetime(view["date"])
        st.line_chart(view.set_index("date")[["forecast_demand"]], height=360)
        cols = st.columns(3)
        cols[0].metric("30-day forecast", f"{view['forecast_demand'].sum():,.0f} units")
        cols[1].metric("Average daily demand", f"{view['forecast_demand'].mean():,.1f} units")
        cols[2].metric("Promo days", f"{int(view['promo_event'].sum())}")
        st.caption("Known future promotions/discounts can be supplied through data/forecast_scenario.csv; unspecified days default to no promotion.")

with tabs[7]:
    st.subheader("Forecast model benchmark")
    if not baseline.empty and not xgb_results.empty:
        benchmark = pd.concat([baseline[["model", "MAE", "RMSE", "MAPE"]], xgb_results[["model", "MAE", "RMSE", "MAPE"]]], ignore_index=True).drop_duplicates(subset=["model"]).sort_values("MAE")
        st.dataframe(benchmark, use_container_width=True, hide_index=True)
    if not backtest.empty:
        st.markdown("### Walk-forward validation")
        st.dataframe(backtest, use_container_width=True, hide_index=True)
        summary = backtest.groupby("model")[["MAE", "RMSE", "MAPE"]].mean().round(3)
        st.markdown("Average metrics across chronological folds")
        st.dataframe(summary, use_container_width=True)

with tabs[8]:
    st.subheader("Decision recommendations")
    st.markdown("1. **Protect revenue:** prioritize critical stockouts and high lost-sales exposure.\n2. **Use the forecast:** replenish against projected demand and lead time.\n3. **Manage suppliers:** investigate deteriorating temporal service signals.\n4. **Compare policies:** use the 30-day replenishment simulation before acting.\n5. **Protect working capital:** balance stockout reduction against incremental holding cost.")
    if not simulation.empty:
        st.markdown("### Highest estimated policy benefit")
        st.dataframe(simulation.sort_values("estimated_net_benefit", ascending=False).head(10), use_container_width=True, hide_index=True)

st.divider()
st.caption("ML-Driven Supply Chain Control Tower — powered by SQL analytics, forecasting, inventory policy, supplier-risk analytics and financial impact simulation.")
