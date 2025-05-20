import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import _tree

# Load shared scaler
scaler = joblib.load('outputs/models/scaler.pkl')

# --- UI Setup ---
st.title("Heart Attack Risk Prediction Dashboard")
st.markdown("A diagnostic support tool powered by Machine Learning and BI insights.")
st.sidebar.header("Enter Patient Data")

# Model selector
model_option = st.sidebar.selectbox("Choose Prediction Model", [
    "Logistic Regression", "Decision Tree", "K-Means Clustering"
])

logistic_features = [
    'Age', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure',
    'Blood sugar', 'CK-MB', 'Troponin', 'pulse_pressure'
]

# NOTE: We will dynamically use the actual model's features below instead of hardcoding.

if model_option == "Logistic Regression":
    # Input sliders for logistic regression
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

    model = joblib.load('outputs/models/logistic_model_final.pkl')
    proba = model.predict_proba(input_scaled)[0][1]

    if proba >= 0.70:
        risk_label = "🔴 High"
    elif proba >= 0.40:
        risk_label = "🟠 Medium"
    else:
        risk_label = "🟢 Low"

    st.subheader(f"Prediction Result ({model_option})")
    st.write("**Heart Attack Risk:**", risk_label)
    st.write(f"**Probability:** {proba:.2%}")

    import plotly.graph_objects as go
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

    st.markdown("🔎 **Clinical Interpretation**")
    if proba >= 0.70:
        st.warning("⚠️ This patient is at **high risk**. Recommend urgent follow-up.")
    elif proba >= 0.40:
        st.info("🩺 Medium risk. Suggest follow-up tests or monitoring.")
    else:
        st.success("✅ Low risk. Encourage healthy lifestyle and regular checkups.")

    coef_dict = dict(zip(logistic_features, model.coef_[0]))
    scaled_vals = input_scaled[0]
    influence = {feat: coef_dict[feat] * scaled_vals[i] for i, feat in enumerate(logistic_features)}

    st.markdown("### 🧠 Top Feature Contributors")
    for feat, val in sorted(influence.items(), key=lambda x: abs(x[1]), reverse=True):
        sign = "↑" if val > 0 else "↓"
        st.write(f"- **{feat}**: {sign} contributed **{val:.2f}**")

elif model_option == "Decision Tree":
    model = joblib.load('outputs/models/decision_tree_model.pkl')
    st.subheader("Step-by-Step Decision Tree Walkthrough")

    tree_ = model.tree_
    all_features = ["Age", "Gender", "Heart rate", "Systolic blood pressure", "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin"]
    feature_names_used = [all_features[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

    node_id = 0
    step = 0
    show_more = True

    while tree_.children_left[node_id] != _tree.TREE_LEAF and show_more:
        feature = feature_names_used[node_id]
        threshold = tree_.threshold[node_id]

        choice = st.radio(
            f"Step {step+1}: Is {feature} ≤ {threshold:.3f}?",
            ["Choose...", "Yes", "No"],
            key=f"node_{node_id}"
        )

        if choice == "Yes":
            node_id = tree_.children_left[node_id]
        elif choice == "No":
            node_id = tree_.children_right[node_id]
        else:
            show_more = False

        step += 1

    if tree_.children_left[node_id] == _tree.TREE_LEAF:
        value = tree_.value[node_id][0]
        prediction = int(np.argmax(value))
        total = int(np.sum(value))
        probability = value[prediction] / total

        st.markdown("---")
        st.markdown("### Final Decision")
        st.write("**Prediction:**", "🔴 Heart Attack" if prediction == 1 else "🟢 No Heart Attack")
        st.write(f"**Confidence:** {probability:.2%} based on {total} samples")

else:
    st.subheader("K-Means Clustering")
    st.info("K-Means visualization coming soon...")