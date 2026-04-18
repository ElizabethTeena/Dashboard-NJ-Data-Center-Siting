import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Objective 1", layout="wide")

st.title(" 📊 Objective 1 – Site Suitability Analysis")
st.markdown("""
- Identified potential sites for sustainable data center development  
- Evaluated sites using a weighted suitability score (MCDA)  
- Considered proximity to substations, transmission lines, and renewable energy sources  
- Presented results through a ranked table and an interactive map for easy comparison  
""")
st.info(
    "Note: Some sites may share the same location but are linked to different substations. "
    "Use the 'Site view' filter to switch between all records and unique site locations."
)
# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/final_mcda_sites1.csv")

df = load_data().copy()

# ----------------------------
# Clean + prepare
# ----------------------------
df = df.dropna(subset=["X", "Y", "MCDA_SCORE", "SITE_SCORE"])
df = df.sort_values("MCDA_SCORE", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1

mcda_max = df["MCDA_SCORE"].max() if df["MCDA_SCORE"].max() != 0 else 1
site_max = df["SITE_SCORE"].max() if df["SITE_SCORE"].max() != 0 else 1

df["mcda_percent"] = ((df["MCDA_SCORE"] / mcda_max) * 100).round(1)
df["site_percent"] = ((df["SITE_SCORE"] / site_max) * 100).round(1)

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filters")

site_view = st.sidebar.radio(
    "Site view",
    ["Show all records", "Show unique site locations only"],
    index=1
)

top_n_option = st.sidebar.selectbox(
    "Show top sites",
    ["Top 50", "Top 100", "Top 200", "All"],
    index=0
)

# Use unique coordinates if selected
if site_view == "Show unique site locations only":
    working_df = (
        df.sort_values("MCDA_SCORE", ascending=False)
          .drop_duplicates(subset=["X", "Y"])
          .reset_index(drop=True)
          .copy()
    )
    working_df["rank"] = working_df.index + 1
else:
    working_df = df.copy()

if top_n_option == "Top 50":
    filtered = working_df.head(50).copy()
elif top_n_option == "Top 100":
    filtered = working_df.head(100).copy()
elif top_n_option == "Top 200":
    filtered = working_df.head(200).copy()
else:
    filtered = working_df.copy()

st.sidebar.write(f"Showing {len(filtered)} sites")

# ----------------------------
# KPI row
# ----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Locations Shown", f"{len(filtered)}")
col2.metric(
    "Highest Suitability Score",
    f"{filtered['MCDA_SCORE'].max():.2f}" if len(filtered) else "N/A"
)
col3.metric(
    "Average Suitability Score",
    f"{filtered['MCDA_SCORE'].mean():.2f}" if len(filtered) else "N/A"
)
# ----------------------------
# Map helpers
# ----------------------------
def marker_color(score_pct: float) -> str:
    if score_pct >= 90:
        return "#1a9850"
    elif score_pct >= 75:
        return "#66bd63"
    elif score_pct >= 60:
        return "#fdae61"
    else:
        return "#d73027"

def build_popup(row) -> str:
    return f"""
    <div style="font-family: Arial; font-size: 14px; line-height: 1.6;">
        <div style="font-size: 16px; font-weight: 700; margin-bottom: 8px;">
            📍 Site Rank #{row['rank']}
        </div>
        <div>🏆 <b>MCDA Score:</b> {row['MCDA_SCORE']:.2f} ({row['mcda_percent']:.1f}%)</div>
        <div>⚡ <b>Site Score:</b> {row['SITE_SCORE']:.2f} ({row['site_percent']:.1f}%)</div>
        <div>📏 <b>Distance:</b> {row['distance']:.2f}</div>
        <div>🔌 <b>TX Distance:</b> {row['tx_distance']:.2f}</div>
        <div>🏭 <b>Substation ID:</b> {row.get('subfull_id', 'N/A')}</div>
        <div>🗂️ <b>OSM ID:</b> {row.get('subosm_id', 'N/A')}</div>
        <div>📌 <b>X:</b> {row['X']:.6f}</div>
        <div>📌 <b>Y:</b> {row['Y']:.6f}</div>
    </div>
    """

def build_tooltip(row) -> str:
    return f"""
    <b>Rank:</b> {row['rank']}<br>
    <b>MCDA:</b> {row['MCDA_SCORE']:.2f}<br>
    <b>Site Score:</b> {row['SITE_SCORE']:.2f}
    """

# ----------------------------
# Build map
# ----------------------------
st.subheader("🗺️ Interactive Site Map")
st.markdown("""
- View the spatial distribution of candidate sites on the below interactive map  
- Each marker represents a site based on its geographic location  
- Clusters help explore densely located areas more easily  
- Marker colors indicate suitability score (higher = more suitable)  
""")
st.markdown("💡 Hover over markers to see site details.")

if len(filtered) == 0:
    st.warning("No sites match the current filters.")
else:
    m = folium.Map(
        location=[39.8, -98.6],
        zoom_start=4,
        tiles="CartoDB positron"
    )

    marker_cluster = MarkerCluster(name="Top Sites").add_to(m)

    for _, row in filtered.iterrows():
        color = marker_color(row["mcda_percent"])

        icon_html = f"""
        <div style="
            background-color: {color};
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
            location=[row["Y"], row["X"]],
            tooltip=folium.Tooltip(build_tooltip(row)),
            popup=folium.Popup(build_popup(row), max_width=380),
            icon=folium.DivIcon(html=icon_html)
        ).add_to(marker_cluster)

    bounds = [
        [filtered["Y"].min(), filtered["X"].min()],
        [filtered["Y"].max(), filtered["X"].max()]
    ]
    m.fit_bounds(bounds)

    legend_html = """
    <div style="
    position: fixed;
    bottom: 40px;
    left: 40px;
    width: 220px;
    background-color: white;
    border: 2px solid #999;
    z-index: 9999;
    font-size: 14px;
    padding: 12px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    ">
    <b>Objective 1 Legend</b><br>
    <span style="color:#1a9850;">●</span> 90–100%<br>
    <span style="color:#66bd63;">●</span> 75–89%<br>
    <span style="color:#fdae61;">●</span> 60–74%<br>
    <span style="color:#d73027;">●</span> Below 60%<br><br>
    Marker number = site rank
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=None, height=650)


# ----------------------------
# Table
# ----------------------------
st.subheader("📋 Site List")
st.markdown("""
- View ranked candidate sites based on suitability scores  
- Sort and compare sites using different criteria  
- Each row shows location, infrastructure proximity, and evaluation scores  
- Final suitability score is scaled to 50, while infrastructure score is scaled to 100  
""")

# ----------------------------
# Sorting controls (above table)
# ----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    sort_label = st.selectbox(
        "Sort by",
        ["Final Suitability Score", "Infrastructure Score"]
    )

sort_column = "MCDA_SCORE" if sort_label == "Final Suitability Score" else "SITE_SCORE"
with col2:
    sort_order = st.radio(
        "Order",
        ["Descending", "Ascending"],
        horizontal=True
    )


table_cols = [
    "rank",
    "X",
    "Y",
    "subfull_id",
    "subosm_id",
    "distance",
    "tx_distance",
    "SITE_SCORE",
    "MCDA_SCORE",
    "site_percent",
    "mcda_percent",
    
    
]

available_cols = [c for c in table_cols if c in filtered.columns]
ascending = True if sort_order == "Ascending" else False

filtered_sorted = filtered.sort_values(
    by=sort_column,
    ascending=ascending
)

column_rename_map = {
    "rank": "Rank",
    "X": "Longitude",
    "Y": "Latitude",
    "subfull_id": "Nearest Substation ID",
    "subosm_id": "Related OSM ID",
    "distance": "Distance to Substation",
    "tx_distance": "Distance to Transmission Line",
    "SITE_SCORE": "Infrastructure Score",
    "MCDA_SCORE": "Final Suitability Score",
    "site_percent": "Infrastructure Score (%)",
    "mcda_percent": "Final Suitability Score (%)",
    
}

display_table = filtered_sorted[available_cols].rename(columns=column_rename_map)

st.dataframe(display_table, use_container_width=True)

# ----------------------------
# Download filtered table
# ----------------------------
csv_data = display_table.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download filtered site list as CSV",
    data=csv_data,
    file_name="objective1_filtered_sites.csv",
    mime="text/csv"
)