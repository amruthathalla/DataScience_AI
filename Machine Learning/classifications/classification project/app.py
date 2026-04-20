import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd

st.set_page_config(page_title="ML Predictor", page_icon="🔮", layout="centered")
st.title("🔮 Load a Trained Model & Predict")

@st.cache_resource
def load_models():
    folder = r"c:\Users\Amrutha Thalla\FSDS\DataScience_AI\Machine Learning\classifications\classification project\models"
    models = {}
    scaler = None
    if not os.path.exists(folder):
        st.error(f"Folder '{folder}' not found. Run backend first.")
        return models, scaler
    for file in os.listdir(folder):
        if file.endswith(".pkl") and file != "scaler.pkl":
            with open(os.path.join(folder, file), "rb") as f:
                name = file.replace(".pkl", "").replace("_", " ")
                models[name] = pickle.load(f)
    scaler_path = os.path.join(folder, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    return models, scaler

models, scaler = load_models()
if not models:
    st.stop()

st.sidebar.success(f"Loaded {len(models)} models")

age = st.number_input("Age", 18, 80, 30)
salary = st.number_input("Estimated Salary", 10000, 200000, 50000)
model_name = st.selectbox("Choose model", list(models.keys()))

if st.button("Predict"):
    if scaler is None:
        st.error("Scaler missing")
    else:
        input_scaled = scaler.transform([[age, salary]])
        pred = models[model_name].predict(input_scaled)[0]
        conf = None
        if hasattr(models[model_name], "predict_proba"):
            conf = max(models[model_name].predict_proba(input_scaled)[0]) * 100
        if pred == 1:
            st.success("✅ Will Purchase")
        else:
            st.error("❌ Will Not Purchase")
        if conf:
            st.metric("Confidence", f"{conf:.1f}%")

if st.button("Compare all models"):
    input_scaled = scaler.transform([[age, salary]])
    data = []
    for name, m in models.items():
        pred = m.predict(input_scaled)[0]
        conf = None
        if hasattr(m, "predict_proba"):
            conf = max(m.predict_proba(input_scaled)[0]) * 100
        data.append({"Model": name, "Prediction": "✅ Purchase" if pred==1 else "❌ No", "Confidence": f"{conf:.1f}%" if conf else "N/A"})
    st.dataframe(pd.DataFrame(data))