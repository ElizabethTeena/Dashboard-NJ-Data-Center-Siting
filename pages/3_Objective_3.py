import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Objective 3", layout="wide")

st.title("🏗️ Objective 3 – Transformer Scenario Cost and Parcel Feasibility Analysis")

st.markdown("""
- Evaluates economic feasibility of candidate parcels  
- Considers different transformer voltage scenarios (120kV, 240kV, 360kV)  
- Combines infrastructure cost and equipment cost  
- Ranks parcels based on cost efficiency  
- Displays results using tables and interactive maps  
""")
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
scenario_costs = {
    "120kV": {
        "transformer": 200000,
        "switchgear": 60000,
        "ups_battery": 300000,
        "pdu": 20000,
        "generator": 120000
    },
    "240kV": {
        "transformer": 500000,
        "switchgear": 90000,
        "ups_battery": 350000,
        "pdu": 20000,
        "generator": 150000
    },
    "360kV": {
        "transformer": 800000,
        "switchgear": 120000,
        "ups_battery": 420000,
        "pdu": 20000,
        "generator": 180000
    }
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
st.sidebar.header("⚙️ Scenario Controls")

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
selected_costs = scenario_costs[scenario]

total_equipment_cost = (
    selected_costs["transformer"] +
    selected_costs["switchgear"] +
    selected_costs["ups_battery"] +
    selected_costs["pdu"] +
    selected_costs["generator"]
)

st.sidebar.markdown("---")
st.sidebar.subheader(f"💰 {scenario} Equipment Cost Assumptions")
st.sidebar.markdown(f"""
- **Transformer:** ${selected_costs['transformer']:,}
- **Switchgear:** ${selected_costs['switchgear']:,}
- **UPS & Battery:** ${selected_costs['ups_battery']:,}
- **PDU:** ${selected_costs['pdu']:,}
- **Generator:** ${selected_costs['generator']:,}
- **Total Equipment Cost:** ${total_equipment_cost:,}
""")



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
st.subheader("🧠 Cost Evaluation Method")

st.markdown("""
- Estimates total development cost for each parcel  
- Calculates grid cost using distance to substation and transmission lines  
- Uses fixed cost per mile for infrastructure connection  
- Adds equipment cost based on selected voltage scenario  
- Ranks parcels using normalized cost-efficiency score (lower cost = higher efficiency)  
""")

st.latex(r"Substation\ Access\ Cost_i = Distance\ to\ Substation_i \times 2{,}290{,}000")
st.latex(r"Transmission\ Connection\ Cost_i = Distance\ to\ Transmission_i \times 2{,}290{,}000")
st.latex(r"Grid\ Cost_i = Substation\ Access\ Cost_i + Transmission\ Connection\ Cost_i")
st.latex(r"Equipment\ Cost_s = Transformer_s + Switchgear_s + UPS/Battery_s + PDU_s + Generator_s")
st.latex(r"Total\ Cost_i = Grid\ Cost_i + Equipment\ Cost_s")
st.latex(r"Cost\ Index_i = \frac{Total\ Cost_i - \min(Total\ Cost)}{\max(Total\ Cost) - \min(Total\ Cost)}")
st.latex(r"Cost\ Efficiency_i = 1 - Cost\ Index_i")

st.markdown("""
- i represents each parcel  
- s represents selected transformer scenario  
- Higher cost-efficiency score indicates more economically favorable parcels  
""")
st.info(
    "Note: Grid-related infrastructure cost is estimated using a fixed cost of $2,290,000 per mile "
    "for both substation access and transmission connection."
)

# ----------------------------
# KPI row
# ----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Selected Voltage Scenario", scenario)
col2.metric("Parcels Displayed", f"{len(filtered)}")
col3.metric("Highest Normalized Cost Efficiency", f"{filtered['cost_efficiency'].max():.4f}" if len(filtered) else "N/A")


# ----------------------------
# Summary table
# ----------------------------
st.subheader("📋 Ranked Parcel Feasibility Table")

st.markdown("""
- Displays top-ranked parcels based on cost efficiency  
- Shows distances to substation and transmission lines  
- Includes grid cost, equipment cost, and total development cost  
- Helps compare parcels under the selected voltage scenario  
""")

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

display_table = filtered[available_cols].rename(columns={
    "rank": "Rank",
    "PAMS_PIN": "Parcel ID",
    "GIS_PIN": "GIS Parcel ID",
    "COUNTY": "County",
    "MUN_NAME": "Municipality",
    "PROP_LOC": "Property Location",
    "dist_to_sub_miles": "Distance to Nearest Substation (miles)",
    "dist_to_trans_miles": "Distance to NearestTransmission Line (miles)",
    "grid_cost": "Estimated Grid Connection Cost ($)",
    "equipment_cost": "Estimated Equipment Cost ($)",
    "total_cost": "Estimated Total Development Cost ($)",
    "cost_efficiency": "Normalized Cost Efficiency Score"
})

st.dataframe(display_table, use_container_width=True)



# ----------------------------
# Map
# ----------------------------
st.subheader("🗺️ Interactive Parcel Feasibility Map")

st.markdown("""
- Shows spatial distribution of top-ranked parcels  
- Each marker represents a parcel location  
- Displays parcel details including cost and distances  
- Helps identify geographically favorable sites  
""")
st.markdown("💡 Hover or click on markers to view parcel details.")

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
                <div>🗂️ <b>GIS Parcel ID:</b> {row.get('GIS_PIN', 'NA')}</div>
                <div>🏠 <b>Property Location:</b> {row.get('PROP_LOC', 'NA')}</div>
                <div>🗺️ <b>County:</b> {row.get('COUNTY', 'NA')}</div>
                <div>🏙️ <b>Municipality:</b> {row.get('MUN_NAME', 'NA')}</div>
                <div>📏 <b>Distance to Substation:</b> {row['dist_to_sub_miles']:.4f} miles</div>
                <div>🔌 <b>Distance to Transmission Line:</b> {row['dist_to_trans_miles']:.4f} miles</div>
                <div>💵 <b>Grid Connection Cost:</b> ${row['grid_cost']:,.2f}</div>
                <div>⚙️ <b>Equipment Cost:</b> ${row['equipment_cost']:,.2f}</div>
                <div>💰 <b>Total Development Cost:</b> ${row['total_cost']:,.2f}</div>
                <div>📊 <b>Normalized Cost Efficiency Score:</b> {row['cost_efficiency']:.4f}</div>
                <div>⚡ <b>Voltage Scenario:</b> {scenario}</div>
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

csv_data = display_table.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered parcel list as CSV",
    data=csv_data,
    file_name=f"objective3_{scenario}_top{top_n}.csv",
    mime="text/csv"
)