import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


# ---------- 模拟训练好的 Ridge 模型（你可以替换成你自己的模型）----------
# 下面这组参数来自你的报告中 HR 的 ridge 回归模型
def predict_hr(age, weight, height, temperature, humidity):
    intercept = 221.93
    coef_temp = 0.1301
    coef_hum = -0.0501
    coef_age = -0.5892
    coef_weight = 0.0643
    coef_height = -0.2303

    return intercept + coef_temp * temperature + coef_hum * humidity + \
           coef_age * age + coef_weight * weight + coef_height * height

def predict_rr(age, weight, height, temperature, humidity):
    intercept = 69.78
    return intercept + 0.0154 * temperature - 0.0087 * humidity - 0.2041 * age - 0.0568 * weight - 0.0899 * height

def predict_vo2max_male(age, weight, height, temperature, humidity):
    intercept = -1367.64
    return intercept - 13.29 * temperature - 1.30 * humidity - 9.78 * age + 18.56 * weight + 23.95 * height

def predict_vo2max_female(age, weight, height, temperature, humidity):
    intercept = -2785.38
    return intercept + 27.80 * temperature + 2.06 * humidity + 2.74 * age + 20.40 * weight + 20.09 * height


# ---------- Streamlit 页面配置 ----------
st.set_page_config(page_title="Athlete Performance Predictor", layout="centered")

st.title("🏃 Athlete Performance Predictor")
st.markdown("Enter your personal and environmental parameters to predict your performance during high-intensity exercise.")

# ---------- 用户输入 ----------
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age (years)", 10, 80, 25)
    sex = st.selectbox("Sex", ("Male", "Female"))
    weight = st.slider("Weight (kg)", 30.0, 120.0, 70.0)
with col2:
    height = st.slider("Height (cm)", 140.0, 200.0, 175.0)
    temperature = st.slider("Ambient Temperature (°C)", 15.0, 35.0, 25.0)
    humidity = st.slider("Humidity (%)", 20.0, 90.0, 50.0)

# ---------- 计算 ----------
hr = predict_hr(age, weight, height, temperature, humidity)
rr = predict_rr(age, weight, height, temperature, humidity)
vo2max = predict_vo2max_male(age, weight, height, temperature, humidity) if sex == "Male" else predict_vo2max_female(age, weight, height, temperature, humidity)

# Anaerobic threshold
max_hr = 220 - age
anaerobic_threshold = 0.8 * max_hr
in_anaerobic_zone = hr > anaerobic_threshold

# ---------- 显示结果 ----------
st.subheader("🔍 Predicted Performance Metrics")
st.write(f"**Predicted Heart Rate (HR):** {hr:.1f} bpm")
st.write(f"**Predicted Respiratory Rate (RR):** {rr:.1f} breaths/min")
st.write(f"**Predicted VO₂max:** {vo2max:.1f} mL/min")

if in_anaerobic_zone:
    st.markdown("🟥 **Warning: Anaerobic zone reached!** Monitor training intensity closely.")
else:
    st.markdown("🟩 **Safe zone: Below anaerobic threshold.**")

# ---------- 附加说明 ----------
st.markdown("---")
st.markdown("📘 This tool uses Ridge Regression models trained on the PhysioNet Treadmill Exercise Dataset. It predicts high-intensity performance responses based on personal and environmental parameters.")


