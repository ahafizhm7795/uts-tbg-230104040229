import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import os

st.set_page_config(
    page_title="Smart Hospital Dashboard",
    layout="wide"
)

st.title("🏥 Smart Hospital Monitoring System")

# =========================
# LOAD PARQUET
# =========================
base_path = os.getcwd()

patient_total_path = f"{base_path}/output/patient_total"
patient_time_path = f"{base_path}/output/patient_time"
ml_data_path = f"{base_path}/output/ml_data"

df_total = pd.read_parquet(patient_total_path)
df_time = pd.read_parquet(patient_time_path)
df_ml = pd.read_parquet(ml_data_path)

# =========================
# SIDEBAR FILTER
# =========================
rooms = df_time["room"].unique()

selected_room = st.sidebar.selectbox(
    "Pilih Ruangan",
    rooms
)

filtered_time = df_time[df_time["room"] == selected_room]

# =========================
# KPI
# =========================
filtered_total = df_total[df_total["room"] == selected_room]

total_patient = int(filtered_total["total_patient"].values[0])

col1, col2 = st.columns(2)

col1.metric(
    "Ruangan",
    selected_room
)

col2.metric(
    "Total Pasien",
    total_patient
)

# =========================
# TREND CHART
# =========================
st.subheader("📈 Tren Pasien")

filtered_time["time_label"] = (
    filtered_time["hour"].astype(str)
    + ":"
    + filtered_time["minute_group"].astype(str)
)

fig = px.line(
    filtered_time,
    x="time_label",
    y="avg_patient",
    markers=True,
    title=f"Trend Pasien - {selected_room}"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# MACHINE LEARNING
# =========================
st.subheader("🤖 Prediksi Jumlah Pasien")

X = df_ml[["hour"]]
y = df_ml["patient_count"]

model = LinearRegression()
model.fit(X, y)

selected_hour = st.slider(
    "Pilih Jam Prediksi",
    0,
    23,
    12
)

prediction = model.predict([[selected_hour]])[0]

st.success(
    f"Prediksi jumlah pasien pada jam {selected_hour}:00 adalah {prediction:.2f}"
)

# =========================
# ANALYSIS
# =========================
st.subheader("📊 Analisis")

highest_room = df_total.sort_values(
    by="total_patient",
    ascending=False
).iloc[0]

st.info(
    f"Ruangan dengan jumlah pasien tertinggi adalah "
    f"{highest_room['room']} "
    f"dengan total {highest_room['total_patient']} pasien."
)