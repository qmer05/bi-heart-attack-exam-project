import streamlit as st
import numpy as np
import joblib

# Shared scaler
scaler = joblib.load('outputs/models/scaler.pkl')

# UI setup
st.title("Heart Attack Risk Prediction Dashboard")
st.markdown("A diagnostic support tool powered by Machine Learning and BI insights.")
st.sidebar.header("Enter Patient Data")

# --- Model Selection ---
model_option = st.sidebar.selectbox("Choose Prediction Model", [
    "Logistic Regression", "Decision Tree", "K-Means Clustering"
])

# --- Input fields ---
age = st.sidebar.slider("Age", 18, 99, 56, format="%d years")
heart_rate = st.sidebar.slider("Heart Rate", 35, 130, 75, format="%d bpm")
sbp = st.sidebar.slider("Systolic Blood Pressure", 60, 200, 130, format="%d mmHg")
dbp = st.sidebar.slider("Diastolic Blood Pressure", 35, 120, 70, format="%d mmHg")
blood_sugar = st.sidebar.slider("Blood Sugar", 30, 280, 120, format="%d mg/dL")
ckmb = st.sidebar.slider("CK-MB", 0.30, 12.99, 2.49, format="%.2f ng/mL")
troponin_ui = st.sidebar.slider("Troponin", 1, 25, 5, format="%d ng/mL")
troponin = troponin_ui / 1000

pulse_pressure = sbp - dbp
input_data = np.array([[age, heart_rate, sbp, dbp, blood_sugar, ckmb, troponin, pulse_pressure]])
input_scaled = scaler.transform(input_data)

# --- Model logic ---
if model_option == "Logistic Regression":
    model = joblib.load('outputs/models/logistic_model_final.pkl')
    proba = model.predict_proba(input_scaled)[0][1]
    if proba >= 0.70:
        risk_label = "🔴 High"
    elif proba >= 0.40:
        risk_label = "🟠 Medium"
    else:
        risk_label = "🟢 Low"
    st.subheader("Prediction Result (Logistic Regression)")
    st.write("**Heart Attack Risk:**", risk_label)
    st.write(f"**Probability:** {proba:.2%}")

elif model_option == "Decision Tree":
    model = joblib.load('outputs/models/decision_tree_model.pkl')
    proba = model.predict_proba(input_scaled)[0][1]
    if proba >= 0.70:
        risk_label = "🔴 High"
    elif proba >= 0.40:
        risk_label = "🟠 Medium"
    else:
        risk_label = "🟢 Low"
    st.subheader("Prediction Result (Decision Tree)")
    st.write("**Heart Attack Risk:**", risk_label)
    st.write(f"**Probability:** {proba:.2%}")

elif model_option == "K-Means Clustering":
    model = joblib.load('outputs/models/kmeans_model.pkl')
    cluster = model.predict(input_scaled)[0]
    st.subheader("Cluster Assignment (K-Means)")
    st.write(f"**This patient belongs to Cluster:** {cluster}")
    st.info("Note: K-Means does not predict heart attack directly, but groups similar profiles.")