import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Supply Chain AI Dashboard", layout="wide")

# ---------------- ADVANCED UI STYLE ---------------- #
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

h1 {
    text-align: center;
    font-size: 48px !important;
    font-weight: 700;
}

.block-container {
    padding-top: 2rem;
}

div.stButton > button {
    font-size: 18px;
    padding: 12px;
    border-radius: 12px;
    background: #1e293b;
    color: white;
    border: 1px solid #334155;
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    background: #2563eb;
    transform: scale(1.05);
    box-shadow: 0px 4px 20px rgba(37,99,235,0.4);
}

[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
}

hr {
    border: 1px solid #334155;
}

section[data-testid="stSidebar"] {
    background: #020617;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '..', 'models', 'model.pkl')

    with open(model_path, 'rb') as f:
        return pickle.load(f)

saved = load_model()
model = saved["model"]
model_columns = saved["columns"]

# ---------------- HEADER ---------------- #
st.markdown("""
<h1>🚚 Supply Chain Risk Intelligence Platform</h1>
<p style='text-align:center; color:#94a3b8;'>
AI-powered late delivery prediction & monitoring system
</p>
""", unsafe_allow_html=True)

# ---------------- NAVIGATION ---------------- #
if "page" not in st.session_state:
    st.session_state.page = "Overview"

col1, col2, col3 = st.columns(3)

def nav_button(label, page_name, col):
    active = st.session_state.page == page_name
    if col.button(label, use_container_width=True):
        st.session_state.page = page_name
    
    if active:
        col.markdown(
            "<div style='text-align:center; color:#3b82f6;'>⬤</div>",
            unsafe_allow_html=True
        )

nav_button("📊 Overview", "Overview", col1)
nav_button("🔍 Prediction", "Prediction", col2)
nav_button("🌍 Analytics", "Analytics", col3)

st.divider()

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("⚙️ Control Panel")

shipping_mode = st.sidebar.selectbox(
    "Shipping Mode", ["Standard Class", "Second Class", "First Class", "Same Day"]
)
customer_segment = st.sidebar.selectbox(
    "Customer Segment", ["Consumer", "Corporate", "Home Office"]
)
market = st.sidebar.selectbox(
    "Market", ["USCA", "LATAM", "Europe", "Asia Pacific"]
)
order_region = st.sidebar.selectbox(
    "Order Region", ["West", "East", "Central", "South"]
)

qty = st.sidebar.slider("Quantity", 1, 50, 2)
days = st.sidebar.slider("Scheduled Days", 1, 10, 3)
discount_rate = st.sidebar.slider("Discount Rate", 0.0, 1.0, 0.1)

sales = st.sidebar.number_input("Sales", value=100.0)
profit = st.sidebar.number_input("Profit", value=20.0)

run = st.sidebar.button("🚀 Run Prediction")

# ---------------- FEATURE ENGINEERING ---------------- #
shipping_pressure = qty / (days + 1)
urgency_index = qty / (days + 1)
discount_impact = discount_rate * qty
profit_ratio = profit / (sales + 1)

# ---------------- DATAFRAME ---------------- #
df = pd.DataFrame({
    "Order Item Quantity": [qty],
    "Days for shipment (scheduled)": [days],
    "Order Item Discount Rate": [discount_rate],
    "Sales": [sales],
    "Order Profit Per Order": [profit],

    "shipping_pressure": [shipping_pressure],
    "urgency_index": [urgency_index],
    "discount_impact": [discount_impact],
    "profit_ratio": [profit_ratio],

    "Shipping Mode": [shipping_mode],
    "Customer Segment": [customer_segment],
    "Market": [market],
    "Order Region": [order_region]
})

df = pd.get_dummies(df)
df = df.reindex(columns=model_columns, fill_value=0)

# ================= OVERVIEW ================= #
if st.session_state.page == "Overview":

    st.subheader("📊 System Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Active Orders", "12,450")
    col2.metric("High Risk Orders", "2,130")
    col3.metric("On-Time Rate", "82%")

    st.markdown("### 📈 Live Insights")

    trend_data = pd.DataFrame({
        "Days": list(range(1, 11)),
        "Risk Score": np.random.uniform(0.3, 0.8, 10)
    })

    fig = px.line(trend_data, x="Days", y="Risk Score")
    st.plotly_chart(fig, use_container_width=True)

# ================= PREDICTION ================= #
elif st.session_state.page == "Prediction":

    st.subheader("🔍 Order Risk Prediction")

    if run:
        with st.spinner("Analyzing risk..."):
            prob = model.predict_proba(df)[0][1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Probability", f"{prob*100:.1f}%")
        col2.metric("Quantity", qty)
        col3.metric("Days", days)

        st.caption(f"Confidence Score: {prob:.2f}")

        if prob < 0.4:
            st.success("🟢 LOW RISK")
            color = "green"
        elif prob < 0.7:
            st.warning("🟡 MEDIUM RISK")
            color = "orange"
        else:
            st.error("🔴 HIGH RISK")
            color = "red"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Risk Level"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

# ================= ANALYTICS ================= #
elif st.session_state.page == "Analytics":

    st.subheader("🌍 Regional Risk Analysis")

    regions = ["West", "East", "Central", "South"]
    risk_values = np.random.uniform(0.3, 0.9, 4)

    fig = px.bar(x=regions, y=risk_values, color=risk_values)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 High Risk Orders")

    table = pd.DataFrame({
        "Order ID": [101, 102, 103, 104],
        "Region": ["West", "East", "Central", "South"],
        "Risk": ["High", "Medium", "High", "Low"]
    })

    st.dataframe(table)