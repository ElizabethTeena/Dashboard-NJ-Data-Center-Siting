import streamlit as st
st.set_page_config(page_title="Sustainable Data Center Siting in NJ", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>👋 Welcome to our Project</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h1 style='text-align: center; color: #2E8B57;''>🌱 Sustainable Data Center Siting in NJ</h1>",
    unsafe_allow_html=True
)
import base64

def get_base64_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()
img_base64 = get_base64_image("home_image.png")

st.markdown(
    f"""
    <div style='text-align: center;'>
        <img src="data:image/png;base64,{img_base64}" 
             style="width:700px; border-radius:12px;">
    </div>
    """,
    unsafe_allow_html=True
)



st.markdown("### Team Members")
st.markdown("""
- 👤 Ankur Patel  
- 👤 Elizabeth Teena Lalu  
- 👤 Venkata Vijay Shankar Passavula  
""")


st.markdown("---")
st.subheader("🎯 Project Objectives")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("## 📊 Objective 1")
    st.write("Site suitability analysis using MCDA scores, ranked site selection, and an interactive map dashboard.")
    if st.button("Open Objective 1", key="obj1"):
        st.switch_page("pages/1_Objective_1.py")

with col2:
    st.markdown("## 🌍 Objective 2")
    st.write("Electricity demand, supply, forecasting, and data center capacity analysis.")
    if st.button("Open Objective 2", key="obj2"):
        st.switch_page("pages/2_Objective_2.py")

with col3:
    st.markdown("## 📍 Objective 3")
    st.write("Transformer scenario cost comparison, parcel ranking, and interactive parcel mapping.")
    if st.button("Open Objective 3", key="obj3"):
        st.switch_page("pages/3_Objective_3.py")
    
    

st.markdown("---")
st.write("👉 Use the sidebar or the buttons above to navigate to each objective page.")