import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Objective 3", layout="wide")

st.title("Objective 3 – Transformer Scenario Cost Dashboard")
st.write(
    "This dashboard compares parcel suitability under 120kV, 240kV, and 360kV "
    "transformer scenarios using total cost and cost efficiency."
)

# ----------------------------
# File paths
# ----------------------------
BASE = "data/O3_Files"

scenario_files = {
    "120kV": f"{BASE}/objective3_120kV.csv",
    "240kV": f"{BASE}/objective3_240kV.csv",
    "360kV": f"{BASE}/objective3_360kV.csv",
}

map_files = {
    "120kV": f"{BASE}/objective3_map_120kV.csv",
    "240kV": f"{BASE}/objective3_map_240kV.csv",
    "360kV": f"{BASE}/objective3_map_360kV.csv",
}

# ----------------------------
# Load helpers
# ----------------------------
@st.cache_data
def load_scenario_table(path):
    return pd.read_csv(path)

@st.cache_data
def load_scenario_map(path):
    return pd.read_csv(path)

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filters")

scenario = st.sidebar.selectbox(
    "Select transformer scenario",
    ["120kV", "240kV", "360kV"],
    index=1
)

top_n = st.sidebar.selectbox(
    "Show top parcels",
    [10, 20, 50, 100],
    index=0
)

# ----------------------------
# Load selected data
# ----------------------------
df = load_scenario_table(scenario_files[scenario]).copy()
map_df = load_scenario_map(map_files[scenario]).copy()

# Clean and sort
df = df.sort_values("cost_efficiency", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1

map_df = map_df.sort_values("cost_efficiency", ascending=False).reset_index(drop=True)
map_df["rank"] = map_df.index + 1

filtered = df.head(top_n).copy()
filtered_map = map_df.head(top_n).copy()

# ----------------------------
# KPI row
# ----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Scenario", scenario)
col2.metric("Top Parcels Shown", f"{len(filtered)}")
col3.metric("Best Cost Efficiency", f"{filtered['cost_efficiency'].max():.4f}" if len(filtered) else "N/A")

# ----------------------------
# Summary table
# ----------------------------
st.subheader("Top Ranked Parcels")

table_cols = [
    "rank",
    "PAMS_PIN",
    "GIS_PIN",
    "COUNTY",
    "MUN_NAME",
    "PROP_LOC",
    "dist_to_sub_miles",
    "dist_to_trans_miles",
    "grid_cost",
    "equipment_cost",
    "total_cost",
    "cost_efficiency",
]

available_cols = [c for c in table_cols if c in filtered.columns]
st.dataframe(filtered[available_cols], use_container_width=True)

# ----------------------------
# Map
# ----------------------------
st.subheader("Interactive Parcel Map")

if len(filtered_map) == 0:
    st.warning("No parcels available for the selected filters.")
else:
    # Ensure lat/lon columns exist
    if "latitude" not in filtered_map.columns or "longitude" not in filtered_map.columns:
        st.error("Map file is missing latitude/longitude columns.")
    else:
        m = folium.Map(
            location=[filtered_map["latitude"].mean(), filtered_map["longitude"].mean()],
            zoom_start=9,
            tiles="CartoDB positron"
        )

        cluster = MarkerCluster(name="Top Parcels").add_to(m)

        for _, row in filtered_map.iterrows():
            popup_text = f"""
            <div style="font-family: Arial; font-size: 14px; line-height: 1.6;">
                <div style="font-size: 16px; font-weight: 700; margin-bottom: 8px;">
                    📍 Parcel Rank #{row['rank']}
                </div>
                <div>🧾 <b>Parcel ID:</b> {row.get('PAMS_PIN', 'NA')}</div>
                <div>🗂️ <b>GIS PIN:</b> {row.get('GIS_PIN', 'NA')}</div>
                <div>🏠 <b>Address:</b> {row.get('PROP_LOC', 'NA')}</div>
                <div>🗺️ <b>County:</b> {row.get('COUNTY', 'NA')}</div>
                <div>🏙️ <b>Municipality:</b> {row.get('MUN_NAME', 'NA')}</div>
                <div>📏 <b>Distance to Substation:</b> {row['dist_to_sub_miles']:.4f} miles</div>
                <div>🔌 <b>Distance to Transmission:</b> {row['dist_to_trans_miles']:.4f} miles</div>
                <div>💰 <b>Total Cost:</b> ${row['total_cost']:,.2f}</div>
                <div>⚡ <b>Cost Efficiency:</b> {row['cost_efficiency']:.4f}</div>
                <div>🏭 <b>Scenario:</b> {scenario}</div>
            </div>
            """

            icon_html = f"""
            <div style="
                background-color: #2E8B57;
                color: white;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                text-align: center;
                font-size: 12px;
                font-weight: bold;
                line-height: 30px;
                border: 2px solid white;
                box-shadow: 0 2px 6px rgba(0,0,0,0.35);
            ">
                {int(row['rank'])}
            </div>
            """

            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_text, max_width=380),
                tooltip=f"{row.get('PAMS_PIN', 'Parcel')} | Rank {row['rank']}",
                icon=folium.DivIcon(html=icon_html)
            ).add_to(cluster)

        st_folium(m, width=None, height=650)

# ----------------------------
# Download
# ----------------------------
csv_data = filtered[available_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered parcel list as CSV",
    data=csv_data,
    file_name=f"objective3_{scenario}_top{top_n}.csv",
    mime="text/csv"
)