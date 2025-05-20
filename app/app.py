import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

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
troponin_ui = st.sidebar.slider("Troponin", 1, 30, 5, format="%d ng/mL")
troponin = troponin_ui / 1000

pulse_pressure = sbp - dbp
input_data = np.array([[age, heart_rate, sbp, dbp, blood_sugar, ckmb, troponin, pulse_pressure]])
input_scaled = scaler.transform(input_data)

# --- Shared logic for both classification models ---
if model_option in ["Logistic Regression", "Decision Tree"]:
    # Load appropriate model
    model_file = 'logistic_model_final.pkl' if model_option == "Logistic Regression" else 'decision_tree_model.pkl'
    model = joblib.load(f'outputs/models/{model_file}')
    
    # Predict probability
    proba = model.predict_proba(input_scaled)[0][1]

    # Risk level label
    if proba >= 0.70:
        risk_label = "🔴 High"
    elif proba >= 0.40:
        risk_label = "🟠 Medium"
    else:
        risk_label = "🟢 Low"

    # --- Section Title ---
    st.subheader(f"Prediction Result ({model_option})")
    st.write("**Heart Attack Risk:**", risk_label)
    st.write(f"**Probability:** {proba:.2%}")

    # --- Risk Gauge Chart ---
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=proba * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "crimson"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': proba * 100
            }
        }
    ))
    st.plotly_chart(fig)

    # --- Contextual Message ---
    st.markdown("🔎 **Clinical Interpretation**")
    if proba >= 0.70:
        st.warning("⚠️ This patient is at **high risk**. Recommend urgent follow-up.")
    elif proba >= 0.40:
        st.info("🩺 Medium risk. Suggest follow-up tests or monitoring.")
    else:
        st.success("✅ Low risk. Encourage healthy lifestyle and regular checkups.")

    # --- Feature Influence (only for Logistic Regression) ---
    if model_option == "Logistic Regression":
        feature_names = ['Age', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure',
                 'Blood sugar', 'CK-MB', 'Troponin', 'pulse_pressure']
        coef_dict = dict(zip(feature_names, model.coef_[0]))   
        scaled_input_flat = input_scaled[0]
        influence = {
            k: coef_dict[k] * scaled_input_flat[i]
            for i, k in enumerate(coef_dict)
        }
        sorted_feats = sorted(influence.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        st.markdown("### 🧠 Top Feature Contributors")
        
        all_feats = sorted(influence.items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, val in all_feats:
            sign = "↑" if val > 0 else "↓"
            st.write(f"- **{feat}**: {sign} contributed **{val:.2f}**")
