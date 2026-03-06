
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Page Config
st.set_page_config(page_title="AI4I Predictive Maintenance Dashboard", layout="wide")

# Title
st.title("🏭 AI4I 2020 Predictive Maintenance Dashboard")
st.markdown("### Data Exploration & Physics-Informed Features Analysis")

# Load Data
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'ai4i2020.csv')
    df = pd.read_csv(data_path)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File `data/ai4i2020.csv` not found. Please ensure the dataset exists.")
    st.stop()

# Sidebar
st.sidebar.header("Controls")
sample_size = st.sidebar.slider("Sample Size (for scatter plots)", 100, len(df), 2000)

# Feature Engineering (Replicating logic for visuals)
df['Power_W'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)
df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Strain_Factor'] = df['Torque [Nm]'] * df['Tool wear [min]']

# Define Failure Type
conditions = [
    (df['TWF'] == 1), (df['HDF'] == 1), (df['PWF'] == 1), (df['OSF'] == 1), (df['RNF'] == 1)
]
choices = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
df['Failure_Type'] = np.select(conditions, choices, default='Normal')

# --- Layout ---

# Row 1: Key Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Samples", len(df))
col2.metric("Total Failures", len(df[df['Machine failure'] == 1]))
col3.metric("Failure Rate", f"{len(df[df['Machine failure'] == 1]) / len(df) * 100:.2f}%")
col4.metric("Avg. Tool Wear", f"{df['Tool wear [min]'].mean():.1f} min")

# Row 2: Failure Distributions
st.subheader("1. Failure Mode Distribution")
fig_dist = px.bar(
    df['Failure_Type'].value_counts().reset_index(), 
    x='Failure_Type', y='count', 
    color='Failure_Type',
    title="Count of Each Failure Mode",
    labels={'Failure_Type': 'Condition', 'count': 'Number of Samples'},
    log_y=True # Important because of imbalance
)
st.plotly_chart(fig_dist, use_container_width=True)
st.caption("*Note: Y-axis is Log Scale due to high class imbalance.*")

with st.expander("ℹ️ Keterangan Failure Mode (Tipe Kegagalan)"):
    st.markdown("""
    * **TWF (Tool Wear Failure)**: Kegagalan akibat keausan alat potong/kerja yang telah melewati waktu batas aman penggunaannya.
    * **HDF (Heat Dissipation Failure)**: Kegagalan akibat mesin *overheating* (suhu terlalu panas). Terjadi jika perbedaan suhu sekitar dan suhu proses sangat rendah diiringi putaran mesin yang tinggi.
    * **PWF (Power Failure)**: Kegagalan daya kelistrikan/mekanik akibat tenaga yang dihasilkan oleh putaran (RPM) dan torsi berada di luar batas rentang daya aman operasional.
    * **OSF (Overstrain Failure)**: Kelebihan beban tarik/tegangan. Sering terjadi ketika gaya beban (torsi) terus-terusan tinggi di tengah kondisi alat potong yang sudah sangat aus.
    * **RNF (Random Failures)**: Kegagalan acak yang terjadi di lapangan. Mencakup insiden operasional yang tidak dapat diprediksi langsung oleh sensor parameter (jarang terjadi).
    """)

# Row 3: Physical Relationships
st.subheader("2. Physics-Informed Analysis")
tab1, tab2, tab3 = st.tabs(["Operational Envelope (Torque vs Speed)", "Tool Wear Analysis", "Heat Analysis"])

with tab1:
    st.markdown("**Power Failure (PWF) Zone:** Look for points where Torque x Speed exceeds design limits.")
    # Filter for faster rendering
    df_sample = df.sample(sample_size, random_state=42)
    # Ensure all failures are included
    df_failures = df[df['Machine failure'] == 1]
    df_viz = pd.concat([df_sample, df_failures]).drop_duplicates()
    
    fig_env = px.scatter(
        df_viz, 
        x='Rotational speed [rpm]', 
        y='Torque [Nm]', 
        color='Failure_Type',
        size=df_viz['Machine failure'].map({0: 2, 1: 8}), # Highlight failures
        hover_data=['Power_W'],
        title="Operational Envelope: Torque vs Speed",
        color_discrete_map={'Normal': 'lightgray', 'PWF': 'red', 'HDF': 'orange', 'OSF': 'blue', 'TWF': 'green', 'RNF': 'purple'}
    )
    # Add Power Limits (Theoretical)
    x_range = np.linspace(df['Rotational speed [rpm]'].min(), df['Rotational speed [rpm]'].max(), 100)
    # P = T * w => T = P / w.  w = rpm * 2pi/60
    # Let's say max power is roughly observed max
    max_p = df[df['PWF']==0]['Power_W'].max()
    
    st.plotly_chart(fig_env, use_container_width=True)

with tab2:
    st.markdown("**Overstrain & Tool Wear:** Interaction between High Torque and High Tool Wear.")
    fig_wear = px.scatter(
        df_viz,
        x='Tool wear [min]',
        y='Torque [Nm]',
        color='Failure_Type',
        title="Tool Wear vs Torque (Strain Analysis)",
         color_discrete_map={'Normal': 'lightgray', 'OSF': 'blue', 'TWF': 'green'}
    )
    st.plotly_chart(fig_wear, use_container_width=True)

with tab3:
    st.markdown("**Heat Dissipation:** Relationship between Process Temperature and Air Temperature.")
    fig_heat = px.scatter(
        df_viz,
        x='Air temperature [K]',
        y='Process temperature [K]',
        color='Failure_Type',
        title="Thermal Map (HDF focus)",
         color_discrete_map={'Normal': 'lightgray', 'HDF': 'orange'}
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# Row 4: Raw Data View
with st.expander("View Raw Data"):
    st.dataframe(df.head(100))

