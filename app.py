import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="WattWise | Energy Insights", layout="wide")
st.markdown("""
<style>
.stApp{background:#f4f7f3;color:#20352d}
.main .block-container{max-width:1180px;padding-top:1.6rem;padding-bottom:3rem}
.hero{background:linear-gradient(120deg,#123c32,#18745d 65%,#c18b35);padding:32px;border-radius:18px;color:white;margin-bottom:22px}
.hero h1{color:white;font-size:36px}.hero p{color:#e4f2e9;font-size:16px}
.section-title{color:#174d3c;font-size:25px;font-weight:750;margin:20px 0 10px}
.metric-card{background:white;border:1px solid #dce8df;border-radius:14px;padding:18px;min-height:125px}
.metric-card h3{color:#a87522;font-size:18px}.metric-card p{color:#53685d;font-size:14px}
.result-card{background:#e7f3eb;border:1px solid #b9d7c3;border-radius:16px;padding:25px;text-align:center}
div.stButton>button[kind="primary"]{background:#167354;border-color:#167354}
footer{visibility:hidden}
</style>
""",unsafe_allow_html=True)

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"electricity_consumption.csv"
MODEL=ROOT/"model"/"electricity_model.pkl"
FEATURES=["Temperature_C","Humidity_Percent","Occupants","AC_Hours","Appliance_Hours","Lighting_Hours","Weekend"]
if "history" not in st.session_state: st.session_state.history=[]
if "latest" not in st.session_state: st.session_state.latest=None

@st.cache_resource
def get_model():
    if MODEL.exists(): return joblib.load(MODEL)
    data=pd.read_csv(DATA)
    m=RandomForestRegressor(n_estimators=160,random_state=42)
    m.fit(data[FEATURES],data["Daily_Consumption_kWh"])
    return m

st.markdown("""<div class="hero"><h1>WattWise</h1>
<p style="font-size:19px"><b>Electricity Consumption Prediction & Energy Insights</b></p>
<p>Estimate daily electricity usage from household conditions and explore practical ways to use energy wisely.</p></div>""",unsafe_allow_html=True)
page=st.radio("Navigation",["Overview","Predict Consumption","Usage History","Energy Tips"],horizontal=True,label_visibility="collapsed")
st.divider()

if page=="Overview":
    st.markdown('<div class="section-title">Energy at a Glance</div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    for col,title,desc in [(a,"Daily Estimate","Predict electricity use in kilowatt-hours (kWh)."),(b,"Household Factors","Consider temperature, occupants and appliance usage."),(c,"Usage History","Review estimates and download your records.")]:
        with col: st.markdown(f'<div class="metric-card"><h3>{title}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">What can influence electricity use?</div>',unsafe_allow_html=True)
    st.bar_chart(pd.DataFrame({"Typical influence":[5,3,2,2]},index=["Air conditioning","Other appliances","Lighting","People in home"]))
    st.caption("Illustrative comparison for explaining the concept, not measured household research data.")
    st.info("Choose Predict Consumption to enter household details and estimate daily electricity use.")

elif page=="Predict Consumption":
    st.markdown('<div class="section-title">Household Details</div>',unsafe_allow_html=True)
    st.write("Enter typical conditions for one day.")
    with st.form("energy_form"):
        name=st.text_input("Household name (optional)",placeholder="Example: Home A")
        c1,c2=st.columns(2)
        with c1:
            temp=st.slider("Outdoor temperature (°C)",16,42,30)
            humidity=st.slider("Humidity (%)",20,90,55)
            occupants=st.number_input("Number of occupants",1,15,4)
            ac=st.slider("Air conditioner use (hours/day)",0.0,12.0,3.0,0.5)
        with c2:
            appliances=st.slider("Other appliance use (combined hours/day)",2.0,18.0,8.0,0.5)
            lighting=st.slider("Lighting use (hours/day)",1.0,12.0,5.0,0.5)
            day=st.selectbox("Day type",["Weekday","Weekend"])
        submit=st.form_submit_button("Estimate Daily Consumption",type="primary",use_container_width=True)
    if submit:
        try:
            model=get_model()
            x=pd.DataFrame([{"Temperature_C":temp,"Humidity_Percent":humidity,"Occupants":occupants,"AC_Hours":ac,"Appliance_Hours":appliances,"Lighting_Hours":lighting,"Weekend":int(day=="Weekend")}])
            pred=max(0,float(model.predict(x)[0]))
            rec={"Date and Time":datetime.now().strftime("%d-%m-%Y %I:%M %p"),"Household":name.strip() or "Not provided","Temperature (°C)":temp,"Humidity (%)":humidity,"Occupants":occupants,"AC Hours":ac,"Appliance Hours":appliances,"Lighting Hours":lighting,"Day Type":day,"Estimated Consumption (kWh)":round(pred,2)}
            st.session_state.history.append(rec); st.session_state.latest=rec
        except Exception as e: st.error(f"Could not generate estimate: {e}")
    if st.session_state.latest:
        r=st.session_state.latest
        st.divider(); st.markdown('<div class="section-title">Estimated Result</div>',unsafe_allow_html=True)
        st.markdown(f"""<div class="result-card"><p style="color:#53685d;font-size:17px">Estimated Daily Electricity Consumption</p><h1 style="color:#167354;font-size:48px">{r['Estimated Consumption (kWh)']:.2f} kWh</h1><p style="color:#53685d">Based on the details entered</p></div>""",unsafe_allow_html=True)
        st.metric("Estimated 30-day usage",f"{r['Estimated Consumption (kWh)']*30:.1f} kWh")
        st.caption("This is an estimate from a synthetic practice dataset, not a utility-meter reading or an electricity bill.")

elif page=="Usage History":
    st.markdown('<div class="section-title">Your Usage Estimates</div>',unsafe_allow_html=True)
    if not st.session_state.history: st.info("No estimates saved in this session yet. Visit Predict Consumption to create one.")
    else:
        hist=pd.DataFrame(st.session_state.history)
        a,b,c=st.columns(3); a.metric("Estimates saved",len(hist)); b.metric("Average daily estimate",f"{hist['Estimated Consumption (kWh)'].mean():.2f} kWh"); c.metric("Highest estimate",f"{hist['Estimated Consumption (kWh)'].max():.2f} kWh")
        st.dataframe(hist,use_container_width=True,hide_index=True)
        st.markdown("### Consumption by estimate")
        st.bar_chart(hist.set_index("Date and Time")["Estimated Consumption (kWh)"])
        st.download_button("Download Usage History (CSV)",hist.to_csv(index=False).encode("utf-8"),"electricity_usage_history.csv","text/csv",use_container_width=True)
        if st.button("Clear session history"):
            st.session_state.history=[]; st.session_state.latest=None; st.rerun()

elif page=="Energy Tips":
    st.markdown('<div class="section-title">Simple Ways to Use Energy Wisely</div>',unsafe_allow_html=True)
    tips=[("Air conditioning","Set a comfortable temperature and keep doors and windows closed while cooling."),("Lighting","Switch off lights when a room is not in use and use daylight where possible."),("Appliances","Turn appliances off when they are not needed; avoid leaving devices on standby."),("Daily habits","Use high-power appliances thoughtfully and check which devices contribute most to usage."),("Track usage","Compare estimates over time and use actual meter readings for household decisions.")]
    for title,detail in tips:
        with st.expander(title): st.write(detail)
    st.warning("These are general suggestions. Actual savings depend on appliances, local conditions and electricity tariffs.")

st.divider()
st.markdown('<div style="text-align:center;color:#718176;padding:12px"><b>WattWise</b><br>Electricity Consumption Insights</div>',unsafe_allow_html=True)
