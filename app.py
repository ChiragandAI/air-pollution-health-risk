"""Streamlit dashboard for the Air Pollution Health Risk Prediction project.

Run after `python3 pipeline.py`:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src import config, data, features

st.set_page_config(page_title="AIRCARE — Pollution Health Risk", layout="wide")


@st.cache_data
def load_df() -> pd.DataFrame:
    return data.load_or_generate()


@st.cache_resource
def load_models() -> dict:
    md = config.MODELS_DIR
    if not (md / "classifier.joblib").exists():
        return {}
    return {
        "clf": joblib.load(md / "classifier.joblib"),
        "reg": joblib.load(md / "regressor.joblib"),
        "kmeans": joblib.load(md / "kmeans.joblib"),
    }


df = load_df()
models = load_models()

st.title("Air Pollution Health Risk — Decision Support Dashboard")
st.caption("Multi-city integrated environmental + weather + location modelling. "
           "Synthetic dataset — see README for scope.")

if not models:
    st.warning("Trained models not found. Run `python3 pipeline.py` first.")
    st.stop()

# ── Sidebar controls ──
cities = sorted(df["city"].unique())
city = st.sidebar.selectbox("City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)
window = st.sidebar.slider("Rolling-mean window (days)", 1, 30, 7)

city_df = df[df["city"] == city].sort_values("date").copy()
city_df["pm25_roll"] = city_df["pm25"].rolling(window, min_periods=1).mean()

# ── Top metrics ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean PM2.5 (µg/m³)", f"{city_df['pm25'].mean():.1f}")
col2.metric("Max PM2.5", f"{city_df['pm25'].max():.0f}")
col3.metric("Mean AQI (proxy)", f"{city_df['aqi'].mean():.0f}")
high_share = (city_df["risk_label"] == "High").mean() * 100
col4.metric("Days in 'High' risk", f"{high_share:.1f} %")

# ── Risk map ──
st.subheader("Risk map — average PM2.5 per city")
city_agg = (df.groupby(["city", "lat", "lon"])
              .agg(pm25_mean=("pm25", "mean"), aqi_mean=("aqi", "mean"))
              .reset_index())
fig_map = px.scatter_mapbox(
    city_agg, lat="lat", lon="lon", size="pm25_mean", color="pm25_mean",
    color_continuous_scale="RdYlGn_r", size_max=40, zoom=3.5,
    hover_name="city", hover_data={"pm25_mean": ":.1f", "aqi_mean": ":.0f"},
    mapbox_style="open-street-map", height=420,
)
fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig_map, use_container_width=True)

# ── Exposure trend ──
st.subheader(f"PM2.5 exposure trend — {city}")
fig_line = px.line(
    city_df, x="date", y=["pm25", "pm25_roll"],
    labels={"value": "PM2.5 (µg/m³)", "date": "Date", "variable": "Series"},
)
st.plotly_chart(fig_line, use_container_width=True)

# ── On-the-fly prediction ──
st.subheader("Predict health-risk category for a custom day")
with st.form("predict"):
    c1, c2, c3 = st.columns(3)
    pm25 = c1.number_input("PM2.5", 5.0, 600.0, 80.0)
    pm10 = c2.number_input("PM10", 8.0, 800.0, 130.0)
    no2 = c3.number_input("NO₂", 2.0, 200.0, 40.0)
    c4, c5, c6 = st.columns(3)
    temp = c4.number_input("Temp °C", -5.0, 50.0, 22.0)
    humidity = c5.number_input("Humidity %", 5.0, 100.0, 60.0)
    wind = c6.number_input("Wind m/s", 0.0, 15.0, 2.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    # Build a single-row feature vector by inheriting yesterday's city profile.
    today = city_df.iloc[-1:].copy()
    today["pm25"] = pm25
    today["pm10"] = pm10
    today["no2"] = no2
    today["temp_c"] = temp
    today["humidity"] = humidity
    today["wind_speed"] = wind
    appended = pd.concat([city_df, today], ignore_index=True)
    X, _, _, _ = features.build_feature_table(appended)
    x_row = X.iloc[[-1]]
    pred_code = int(models["clf"].predict(x_row)[0])
    pred_label = config.RISK_LABELS[pred_code]
    aqi_pred = float(models["reg"].predict(x_row)[0])
    st.success(f"Predicted risk: **{pred_label}** &nbsp;·&nbsp; "
               f"Predicted AQI proxy: **{aqi_pred:.0f}**")

# ── Pipeline artifacts ──
st.subheader("Model interpretability")
out = Path(config.OUTPUTS_DIR)
imgs = [("feature_importance.png", "Top features"),
        ("shap_summary.png", "SHAP mean-|value| per feature"),
        ("confusion_matrix.png", "Confusion matrix (held-out)")]
cols = st.columns(len(imgs))
for col, (fname, caption) in zip(cols, imgs):
    p = out / fname
    if p.exists():
        col.image(str(p), caption=caption, use_container_width=True)
    else:
        col.info(f"{fname} not found — re-run pipeline.")
