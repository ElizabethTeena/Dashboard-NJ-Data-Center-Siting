from prophet import Prophet
import streamlit as st
import pandas as pd
import glob
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Objective 2 Dashboard", layout="wide")

st.title("Objective 2 - REDUCE ENVIRONMENTAL IMPACT")
st.write("Daily electricity demand, supply, and supply-demand gap.")

# ----------------------------
# File paths (LOCAL on your Mac)
# ----------------------------
BASE = "data/O2_Files"

LOAD_DIR = f"{BASE}/hrl_load_estimated(2020-2025)"
GEN_DIR  = f"{BASE}/Generation_FuelType(2020-2025)"
NET_DIR  = f"{BASE}/NetImport(2020-2025)"
PEAK_CSV = f"{BASE}/O2_Load_AnnualPeak_PSEG.csv"

load_files = sorted(glob.glob(f"{LOAD_DIR}/*.csv"))
gen_files  = sorted(glob.glob(f"{GEN_DIR}/*.csv"))
net_files  = sorted(glob.glob(f"{NET_DIR}/*.csv"))

# ----------------------------
# Check files exist
# ----------------------------
st.subheader("File Check")
st.write("Load files found:", len(load_files))
st.write("Generation files found:", len(gen_files))
st.write("Net import files found:", len(net_files))

# ----------------------------
# Load and clean data
# ----------------------------
@st.cache_data
def load_all_data():
    # LOAD
    load = pd.concat([pd.read_csv(f) for f in load_files], ignore_index=True)
    load["dt"] = pd.to_datetime(load["datetime_beginning_ept"], errors="coerce")

    if "load_area" in load.columns:
        load = load[load["load_area"] == "PJME"].copy()

    load = (
        load[["dt", "estimated_load_hourly"]]
        .rename(columns={"estimated_load_hourly": "load_mw"})
        .dropna()
        .sort_values("dt")
    )

    # GENERATION
    gen = pd.concat([pd.read_csv(f) for f in gen_files], ignore_index=True)
    gen["dt"] = pd.to_datetime(
        gen["datetime_beginning_ept"].astype(str).str.strip(),
        format="mixed",
        errors="coerce"
    )

    gen = (
        gen[["dt", "fuel_type", "mw", "is_renewable"]]
        .dropna()
        .sort_values("dt")
    )

    # NET IMPORTS
    net = pd.concat([pd.read_csv(f) for f in net_files], ignore_index=True)
    net["dt"] = pd.to_datetime(net["datetime_beginning_ept"], errors="coerce")

    if "state" in net.columns:
        net = net[net["state"] == "NJ"].copy()

    net = (
        net[["dt", "net_interchange"]]
        .rename(columns={"net_interchange": "net_mw"})
        .dropna()
        .sort_values("dt")
    )

    # PEAK
    peak = pd.read_csv(PEAK_CSV)

    return load, gen, net, peak

load, gen, net, peak = load_all_data()

# ----------------------------
# Daily demand
# ----------------------------
daily_demand = (
    load.set_index("dt")["load_mw"]
    .resample("D")
    .mean()
    .reset_index()
    .rename(columns={"dt": "ds", "load_mw": "demand_mw"})
)

# ----------------------------
# Daily supply
# ----------------------------
gen_hourly_total = (
    gen.groupby("dt", as_index=False)["mw"]
    .sum()
    .rename(columns={"mw": "gen_mw"})
)

gen_daily = (
    gen_hourly_total.set_index("dt")["gen_mw"]
    .resample("D")
    .mean()
    .reset_index()
    .rename(columns={"dt": "ds"})
)

net_daily = (
    net.set_index("dt")["net_mw"]
    .resample("D")
    .mean()
    .reset_index()
    .rename(columns={"dt": "ds"})
)

daily_supply = gen_daily.merge(net_daily, on="ds", how="inner")
daily_supply["supply_mw"] = daily_supply["gen_mw"] + daily_supply["net_mw"]

# ----------------------------
# Merge demand + supply
# ----------------------------
daily = daily_demand.merge(
    daily_supply[["ds", "gen_mw", "net_mw", "supply_mw"]],
    on="ds",
    how="inner"
)

daily["gap_mw"] = daily["supply_mw"] - daily["demand_mw"]

# ----------------------------
# Forecast demand
# ----------------------------
@st.cache_data
def build_demand_forecast(daily_df):
    demand_prophet = daily_df[["ds", "demand_mw"]].rename(columns={"demand_mw": "y"}).copy()

    m_demand = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    m_demand.fit(demand_prophet)

    future_demand = m_demand.make_future_dataframe(periods=365*5, freq="D")
    forecast_demand = m_demand.predict(future_demand)

    demand_forecast_clean = forecast_demand[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    last_date = daily_df["ds"].max()
    future_demand_only = demand_forecast_clean[demand_forecast_clean["ds"] > last_date].copy()

    return forecast_demand, future_demand_only


# ----------------------------
# Forecast supply
# ----------------------------
@st.cache_data
def build_supply_forecast(daily_df):
    supply_prophet = daily_df[["ds", "supply_mw"]].rename(columns={"supply_mw": "y"}).copy()

    m_supply = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    m_supply.fit(supply_prophet)

    future_supply = m_supply.make_future_dataframe(periods=365*5, freq="D")
    forecast_supply = m_supply.predict(future_supply)

    supply_forecast_clean = forecast_supply[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    last_date = daily_df["ds"].max()
    future_supply_only = supply_forecast_clean[supply_forecast_clean["ds"] > last_date].copy()

    return forecast_supply, future_supply_only


forecast_demand, future_demand_only = build_demand_forecast(daily)
forecast_supply, future_supply_only = build_supply_forecast(daily)


# ----------------------------
# Future gap scenarios
# ----------------------------
gap_scen = future_supply_only[["ds"]].copy()
gap_scen["gap_low_mw"] = future_supply_only["yhat"].values - future_demand_only["yhat_lower"].values
gap_scen["gap_med_mw"] = future_supply_only["yhat"].values - future_demand_only["yhat"].values
gap_scen["gap_high_mw"] = future_supply_only["yhat"].values - future_demand_only["yhat_upper"].values

gap_monthly = gap_scen.set_index("ds").resample("ME").mean().reset_index()


# ----------------------------
# Data center capacity calculation
# ----------------------------
dc_capacity = gap_monthly.copy()

dc_capacity["surplus_low"] = dc_capacity["gap_low_mw"].clip(lower=0)
dc_capacity["surplus_med"] = dc_capacity["gap_med_mw"].clip(lower=0)
dc_capacity["surplus_high"] = dc_capacity["gap_high_mw"].clip(lower=0)

# Assuming 30 MW per data center
MW_PER_DC = 30

dc_capacity["dc_low"] = np.floor(dc_capacity["surplus_low"] / MW_PER_DC)
dc_capacity["dc_med"] = np.floor(dc_capacity["surplus_med"] / MW_PER_DC)
dc_capacity["dc_high"] = np.floor(dc_capacity["surplus_high"] / MW_PER_DC)

# PJME peak by year from hourly load
load_tmp = load.copy()
load_tmp["year"] = load_tmp["dt"].dt.year
pjme_peak_by_year = load_tmp.groupby("year")["load_mw"].max().reset_index(name="pjme_peak_mw")

# PSEG peak by year from annual peak file
pseg_peak_by_year = peak[["year", "nspl_mw"]].rename(columns={"nspl_mw": "pseg_peak_mw"}).copy()

# Compute NJ share
share = pd.merge(pseg_peak_by_year, pjme_peak_by_year, on="year", how="inner")
share["nj_share"] = share["pseg_peak_mw"] / share["pjme_peak_mw"]

avg_nj_share = share["nj_share"].mean()

# Add NJ-scaled DC capacity
dc_capacity["ds"] = pd.to_datetime(dc_capacity["ds"], errors="coerce")
dc_capacity["year"] = dc_capacity["ds"].dt.year
dc_capacity["nj_dc_low"] = (dc_capacity["dc_low"] * avg_nj_share).astype(int)
dc_capacity["nj_dc_med"] = (dc_capacity["dc_med"] * avg_nj_share).astype(int)
dc_capacity["nj_dc_high"] = (dc_capacity["dc_high"] * avg_nj_share).astype(int)

# ----------------------------
# Sidebar date filter
# ----------------------------
st.sidebar.header("Filters")

min_date = daily["ds"].min().date()
max_date = daily["ds"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_daily = daily[
        (daily["ds"].dt.date >= start_date) &
        (daily["ds"].dt.date <= end_date)
    ].copy()
else:
    filtered_daily = daily.copy()

# ----------------------------
# Summary metrics
# ----------------------------
st.subheader("Summary")
col1, col2, col3 = st.columns(3)

col1.metric("Average Demand (MW)", f"{filtered_daily['demand_mw'].mean():,.0f}")
col2.metric("Average Supply (MW)", f"{filtered_daily['supply_mw'].mean():,.0f}")
col3.metric("Average Gap (MW)", f"{filtered_daily['gap_mw'].mean():,.0f}")

# ----------------------------
# Charts
# ----------------------------
fig1 = px.line(
    filtered_daily,
    x="ds",
    y="demand_mw",
    title="Daily Electricity Demand (MW)",
    labels={"ds": "Date", "demand_mw": "Demand (MW)"}
)
fig1.update_traces(
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Demand (MW): %{y:,.0f}<extra></extra>"
)
fig1.update_layout(hovermode="x unified", template="plotly_white")

fig2 = px.line(
    filtered_daily,
    x="ds",
    y="supply_mw",
    title="Daily Supply Proxy (MW)",
    labels={"ds": "Date", "supply_mw": "Supply (MW)"}
)
fig2.update_traces(
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Supply (MW): %{y:,.0f}<extra></extra>"
)
fig2.update_layout(hovermode="x unified", template="plotly_white")

fig3 = px.line(
    filtered_daily,
    x="ds",
    y="gap_mw",
    title="Daily Supply-Demand Gap (MW)",
    labels={"ds": "Date", "gap_mw": "Gap (MW)"}
)
fig3.update_traces(
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Gap (MW): %{y:,.0f}<extra></extra>"
)
fig3.add_hline(y=0)
fig3.update_layout(hovermode="x unified", template="plotly_white")

st.subheader("Charts")

st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)
st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# Forecast charts
# ----------------------------
st.subheader("Forecast Charts")

fig_d = px.line(
    forecast_demand,
    x="ds",
    y="yhat",
    title="Forecasted Electricity Demand (MW)",
    labels={"ds": "Date", "yhat": "Demand Forecast (MW)"}
)
fig_d.update_traces(
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast (MW): %{y:,.0f}<extra></extra>"
)
fig_d.update_layout(template="plotly_white", hovermode="x unified")

fig_s = px.line(
    forecast_supply,
    x="ds",
    y="yhat",
    title="Forecasted Electricity Supply (MW)",
    labels={"ds": "Date", "yhat": "Supply Forecast (MW)"}
)
fig_s.update_traces(
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast (MW): %{y:,.0f}<extra></extra>"
)
fig_s.update_layout(template="plotly_white", hovermode="x unified")

plot_df = gap_scen.melt(
    id_vars="ds",
    value_vars=["gap_low_mw", "gap_med_mw", "gap_high_mw"],
    var_name="scenario",
    value_name="gap_mw"
)

fig_gap = px.line(
    plot_df,
    x="ds",
    y="gap_mw",
    color="scenario",
    title="Future Supply-Demand Gap (Low / Medium / High Demand Scenarios)",
    labels={"ds": "Date", "gap_mw": "Gap (MW)", "scenario": "Scenario"}
)
fig_gap.add_hline(y=0)
fig_gap.update_traces(
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Gap (MW): %{y:,.0f}<extra></extra>"
)
fig_gap.update_layout(template="plotly_white", hovermode="x unified")

plot_df_m = gap_monthly.melt(
    id_vars="ds",
    value_vars=["gap_low_mw", "gap_med_mw", "gap_high_mw"],
    var_name="scenario",
    value_name="gap_mw"
)

fig_gap_m = px.line(
    plot_df_m,
    x="ds",
    y="gap_mw",
    color="scenario",
    title="Future Supply-Demand Gap (Monthly Avg) — Low/Med/High Demand Scenarios",
    labels={"ds": "Month", "gap_mw": "Gap (MW)", "scenario": "Scenario"}
)
fig_gap_m.add_hline(y=0)
fig_gap_m.update_traces(
    hovertemplate="Month: %{x|%Y-%m}<br>Gap (MW): %{y:,.0f}<extra></extra>"
)
fig_gap_m.update_layout(template="plotly_white", hovermode="x unified")

st.plotly_chart(fig_d, use_container_width=True)
st.plotly_chart(fig_s, use_container_width=True)
st.plotly_chart(fig_gap, use_container_width=True)
st.plotly_chart(fig_gap_m, use_container_width=True)

# ----------------------------
# Show data table
# ----------------------------
st.subheader("Filtered Data")
st.dataframe(filtered_daily)

# ----------------------------
# PJME vs NJ-scaled DC capacity table
# ----------------------------
st.subheader("PJME vs NJ-scaled DC Capacity")

dc_display = dc_capacity[[
    "year", "ds",
    "dc_low", "nj_dc_low",
    "dc_med", "nj_dc_med",
    "dc_high", "nj_dc_high"
]]

st.dataframe(dc_display)