from prophet import Prophet
import streamlit as st
import pandas as pd
import glob
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Objective 2 Dashboard", layout="wide")
st.title("⚡ Objective 2 – Electricity Demand Forecasting and Capacity Assessment")

st.markdown("""
- Estimated future electricity demand and supply for New Jersey using historical data  
- Applied Prophet  time-series model to forecast energy trends for 2026–2030  
- Calculated supply–demand gap under different growth scenarios (low, medium, high)  
- Used surplus energy to estimate the number of data centers that can be supported  
- Scaled results to New Jersey level using peak load ratio (PJME  to NJ)  
""")

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
# Normalize historical series
# ----------------------------
daily["demand_norm"] = daily["demand_mw"] / daily["demand_mw"].mean()
daily["supply_norm"] = daily["supply_mw"] / daily["supply_mw"].mean()

# Gap can be tricky because its mean may be very close to 0.
# Using absolute mean is safer for display.
gap_scale = daily["gap_mw"].abs().mean()
daily["gap_norm"] = daily["gap_mw"] / gap_scale if gap_scale != 0 else 0
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

# ----------------------------
# Sidebar date filter
# ----------------------------
st.sidebar.header("Filters")
st.sidebar.markdown("### 📅 Historical Date Filter")

min_date = daily["ds"].min().date()
max_date = daily["ds"].max().date()
start_date = st.sidebar.date_input(
    "Start date",
    min_value=min_date,
    max_value=max_date,
    value=min_date
)

end_date = st.sidebar.date_input(
    "End date",
    min_value=min_date,
    max_value=max_date,
    value=max_date
)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date")
filtered_daily = daily[
        (daily["ds"].dt.date >= start_date) &
        (daily["ds"].dt.date <= end_date)
    ].copy()

st.sidebar.markdown("### 🔮 Forecast Date Filter")

forecast_min_date = future_demand_only["ds"].min().date()
forecast_max_date = future_demand_only["ds"].max().date()

forecast_start_date = st.sidebar.date_input(
    "Forecast start date",
    min_value=forecast_min_date,
    max_value=forecast_max_date,
    value=forecast_min_date,
    key="forecast_start_date"
)

forecast_end_date = st.sidebar.date_input(
    "Forecast end date",
    min_value=forecast_min_date,
    max_value=forecast_max_date,
    value=forecast_max_date,
    key="forecast_end_date"
)

if forecast_start_date > forecast_end_date:
    st.sidebar.error("Forecast start date must be before forecast end date")
filtered_future_demand = future_demand_only[
    (future_demand_only["ds"].dt.date >= forecast_start_date) &
    (future_demand_only["ds"].dt.date <= forecast_end_date)
].copy()

filtered_future_supply = future_supply_only[
    (future_supply_only["ds"].dt.date >= forecast_start_date) &
    (future_supply_only["ds"].dt.date <= forecast_end_date)
].copy()

filtered_gap_scen = gap_scen[
    (gap_scen["ds"].dt.date >= forecast_start_date) &
    (gap_scen["ds"].dt.date <= forecast_end_date)
].copy()


# Add NJ-scaled DC capacity
dc_capacity["ds"] = pd.to_datetime(dc_capacity["ds"], errors="coerce")
dc_capacity["year"] = dc_capacity["ds"].dt.year
dc_capacity["nj_dc_low"] = (dc_capacity["dc_low"] * avg_nj_share).astype(int)
dc_capacity["nj_dc_med"] = (dc_capacity["dc_med"] * avg_nj_share).astype(int)
dc_capacity["nj_dc_high"] = (dc_capacity["dc_high"] * avg_nj_share).astype(int)

filtered_dc_capacity = dc_capacity[
    (dc_capacity["ds"].dt.date >= forecast_start_date) &
    (dc_capacity["ds"].dt.date <= forecast_end_date)
].copy()
st.sidebar.markdown("### 📂 Section Navigation")

section_choice = st.sidebar.selectbox(
    "Choose section",
    [
        "Show All",
        "Forecasting Method",
        "Historical Trends",
        "Historical Summary Table",
        "Growth Scenarios & Forecasted Trends",
        "Capacity Estimation & Calculation",
        "NJ Scaling",
        "Final Capacity Table"
    ],
    index=0
)





if section_choice in ["Show All", "Forecasting Method"]:
	st.subheader("🧠 Forecasting Method")

	st.markdown("""
	- Derived daily electricity demand and supply from historical data  
	- Calculated total supply using generation and net imports  
	- Estimated supply–demand gap  
	- Applied Prophet model to forecast demand and supply for 5 years  
	""")

	st.latex(r"Supply_t = Generation_t + NetImports_t")
	st.latex(r"Gap_t = Supply_t - Demand_t")



# ----------------------------
# Summary metrics
# ----------------------------

	st.subheader("📌 Key Energy Indicators")
	st.markdown("""
	- Snapshot of average demand, supply, and gap  
	- Helps understand overall energy balance  
	""")

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

#-----------------#
#-----------------#

if section_choice in ["Show All", "Historical Trends"]:

    st.subheader("📊 Historical Demand, Supply, and Gap Trends")
    st.markdown("""
    - Historical demand, supply, and gap normalized for trend comparison  
    - Each series is scaled by its mean so the average equals 1  
    - Separate charts make each trend easier to read  
    """)

    fig_demand_norm = px.line(
        filtered_daily,
        x="ds",
        y="demand_norm",
        title="Normalized Historical Electricity Demand",
        labels={"ds": "Date", "demand_norm": "Normalized Demand (Mean = 1)"}
    )
    fig_demand_norm.update_traces(
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>"
    )
    fig_demand_norm.add_hline(y=1, line_dash="dash")

    fig_supply_norm = px.line(
        filtered_daily,
        x="ds",
        y="supply_norm",
        title="Normalized Historical Electricity Supply",
        labels={"ds": "Date", "supply_norm": "Normalized Supply (Mean = 1)"}
    )
    fig_supply_norm.update_traces(
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>"
    )
    fig_supply_norm.add_hline(y=1, line_dash="dash")

    fig_gap_norm = px.line(
        filtered_daily,
        x="ds",
        y="gap_norm",
        title="Normalized Historical Supply–Demand Gap",
        labels={"ds": "Date", "gap_norm": "Normalized Gap"}
    )
    fig_gap_norm.update_traces(
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>"
    )
    fig_gap_norm.add_hline(y=0, line_dash="dash")

    st.plotly_chart(fig_demand_norm, use_container_width=True)
    st.plotly_chart(fig_supply_norm, use_container_width=True)
    st.plotly_chart(fig_gap_norm, use_container_width=True)
    
# ----------------------------
# Show data table
# ----------------------------
if section_choice in ["Show All", "Historical Summary Table"]:
	st.subheader("📋 Historical Daily Energy Summary")
	st.markdown("""
	- Displays daily demand, supply, and gap values  
	- Used as input for forecasting analysis  
	""")

	filtered_daily_display = filtered_daily.rename(columns={
    	"ds": "Date",
    	"demand_mw": "Electricity Demand (MW)",
    	"gen_mw": "Electricity Generation (MW)",
    	"net_mw": "Net Imports (MW)",
    	"supply_mw": "Total Supply (MW)",
    	"gap_mw": "Supply–Demand Gap (MW)"
	})

	st.dataframe(filtered_daily_display, use_container_width=True)
if section_choice in ["Show All", "Growth Scenarios & Forecasted Trends"]:
	st.subheader("📈 Future Demand Growth Scenarios")



	st.markdown("""
	Three demand growth scenarios considered to evaluate future uncertainty 
	- Low: 0.5% growth  
	- Medium: 1.5% growth  
	- High: 3.0% growth 
	""")


# ----------------------------
# Forecast charts
# ----------------------------

	st.subheader("🔮 Forecasted Energy Trends (2026–2030)")
	st.markdown("""
	- Shows predicted demand and supply trends  
	- Displays future supply–demand gap  
	- Used to assess long-term energy feasibility  
	""")
	fig_d = px.line(
    	filtered_future_demand,
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
    	filtered_future_supply,
   	x="ds",
    	y="yhat",
    	title="Forecasted Electricity Supply (MW)",
    	labels={"ds": "Date", "yhat": "Supply Forecast (MW)"}
	)
	fig_s.update_traces(
    	hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast (MW): %{y:,.0f}<extra></extra>"
	)
	fig_s.update_layout(template="plotly_white", hovermode="x unified")

	plot_df = filtered_gap_scen.melt(
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

if section_choice in ["Show All", "Capacity Estimation & Calculation"]:
	st.subheader("🏗️ Data Center Capacity Estimation")
	st.markdown("""
	- Uses surplus energy to estimate data center capacity  
	- Only positive gap (surplus) is considered  
	- Each data center assumed to require 30 MW  
	""")

	st.latex(r"Surplus_t = \max(0, Gap_t)")
	st.latex(r"DC_t = \left\lfloor \frac{Surplus_t}{P_{DC}} \right\rfloor")

	st.markdown("""
	- Calculation performed for low, medium, and high scenarios  
	""")

	st.latex(r"DC_{low,t} = \left\lfloor \frac{Surplus_{low,t}}{30} \right\rfloor")
	st.latex(r"DC_{med,t} = \left\lfloor \frac{Surplus_{med,t}}{30} \right\rfloor")
	st.latex(r"DC_{high,t} = \left\lfloor \frac{Surplus_{high,t}}{30} \right\rfloor")
	st.subheader("🧮 How Data Centers Are Calculated")

	st.markdown("""
	- Demonstrates conversion from surplus energy to data centers  
	- Values are divided by 30 MW and rounded down  
	""")

	dc_sample = dc_capacity[[
    	"ds",
    	"surplus_low",
    	"surplus_med",
   	 "surplus_high",
    	"dc_low",
    	"dc_med",
    	"dc_high"
	]].rename(columns={
    	"ds": "Date",
    	"surplus_low": "Available Surplus (Low Scenario)",
    	"surplus_med": "Available Surplus (Medium Scenario)",
    	"surplus_high": "Available Surplus (High Scenario)",
    	"dc_low": "Estimated Data Centers (Low)",
    	"dc_med": "Estimated Data Centers (Medium)",
    	"dc_high": "Estimated Data Centers (High)"
	})

	st.dataframe(dc_sample.head(10), use_container_width=True)

if section_choice in ["Show All", "NJ Scaling"]:
	st.subheader("📑 Peak Load Ratio Used for New Jersey Scaling")
	st.markdown("""
	- Compares PJME peak load with PSEG (NJ) peak load  
	- Used to estimate New Jersey's share of total capacity  
	""")



	share_display = share.rename(columns={
    	"year": "Year",
    	"pseg_peak_mw": "PSEG Peak Load (MW)",
    	"pjme_peak_mw": "PJME Peak Load (MW)",
    	"nj_share": "New Jersey Share of PJME"
	})

	share_display["New Jersey Share of PJME"] = share_display["New Jersey Share of PJME"].round(4)

	st.dataframe(share_display, use_container_width=True)

	st.info(f"Average New Jersey share used for scaling: {avg_nj_share:.4f}")


	st.markdown("""
	- Assumed each data center requires **30 MW** of power  
	- Estimated total data center capacity using available surplus energy  
	- Converted regional (PJME) capacity to New Jersey capacity  
	- Used ratio of NJ peak demand (PSEG) to PJME peak demand  
	- Averaged this ratio across years to get NJ share  
	- Scaled PJME capacity using this share to get NJ-specific estimates  
	""")

	st.latex(r"NJ\ Share_y = \frac{PSEG\ Peak_y}{PJME\ Peak_y}")
	st.latex(r"Average\ NJ\ Share = \frac{1}{n}\sum_{y=1}^{n} NJ\ Share_y")
	st.latex(r"NJ\ DC\ Capacity_t = PJME\ DC\ Capacity_t \times Average\ NJ\ Share")






# ----------------------------
# PJME vs NJ-scaled DC capacity table
# ----------------------------
if section_choice in ["Show All", "Final Capacity Table"]:
	st.subheader("🏢 Estimated Data Center Support Capacity")
	st.markdown("""
	- Shows estimated data center capacity under all scenarios  
	- Includes both PJME and New Jersey estimates  
	- Helps compare regional vs state-level capacity  
	""")
	dc_display = dc_capacity[[
    	"year", "ds",
    	"dc_low", "nj_dc_low",
    	"dc_med", "nj_dc_med",
    	"dc_high", "nj_dc_high"
	]].rename(columns={
    	"year": "Year",
    	"ds": "Month",
    	"dc_low": "PJME Capacity (Low Growth – 0.5%)",
    	"nj_dc_low": "NJ Capacity (Low Growth – 0.5%)",
    	"dc_med": "PJME Capacity (Medium Growth – 1.5%)",
     	"nj_dc_med": "NJ Capacity (Medium Growth – 1.5%)",
    	"dc_high": "PJME Capacity (High Growth – 3.0%)",
    	"nj_dc_high": "NJ Capacity (High Growth – 3.0%)"
	})

	st.dataframe(dc_display, use_container_width=True)