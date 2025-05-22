import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import _tree
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load models & scalers ---
scaler = joblib.load('outputs/models/scaler.pkl')                 # For Logistic Regression
scaler_kmeans = joblib.load('outputs/models/scaler_kmeans.pkl')   # Trained on 7 features only
kmeans = joblib.load('outputs/models/kmeans_model.pkl')           # Trained K-Means model
cluster_profiles = pd.read_csv('outputs/models/cluster_feature_means.csv')
cluster_outcomes = pd.read_csv('outputs/models/cluster_outcome_ratios.csv')

# --- UI Setup ---
st.title("Heart Attack Risk Prediction Dashboard")
st.markdown("A diagnostic support tool powered by Machine Learning and BI insights.")
st.sidebar.header("Enter Patient Data")

# Model selector
model_option = st.sidebar.selectbox("Choose Prediction Model", [
    "Logistic Regression", "Decision Tree", "K-Means Clustering"
])

# Input sliders (shared)
age = st.sidebar.slider("Age", 18, 99, 56, format="%d years")
heart_rate = st.sidebar.slider("Heart Rate", 35, 130, 75, format="%d bpm")
sbp = st.sidebar.slider("Systolic Blood Pressure", 60, 200, 130, format="%d mmHg")
dbp = st.sidebar.slider("Diastolic Blood Pressure", 35, 120, 70, format="%d mmHg")
blood_sugar = st.sidebar.slider("Blood Sugar", 30, 280, 120, format="%d mg/dL")
ckmb = st.sidebar.slider("CK-MB", 0.30, 12.99, 2.49, format="%.2f ng/mL")
# Troponin slider (logistic-friendly: 1-30 ng/mL)
troponin_ui = st.sidebar.slider("Troponin", 1, 30, 5, format="%d ng/mL")
troponin_for_logistic = troponin_ui / 1000       # e.g., 10 -> 0.010 for logistic
troponin_for_kmeans = troponin_ui / 100          # e.g., 10 -> 0.10 for k-means
pulse_pressure = sbp - dbp

# --- Logistic Regression ---
if model_option == "Logistic Regression":
    logistic_features = [
        'Age', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure',
        'Blood sugar', 'CK-MB', 'Troponin', 'pulse_pressure'
    ]
    input_data = np.array([[age, heart_rate, sbp, dbp, blood_sugar, ckmb, troponin_for_logistic, pulse_pressure]])
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

# --- Decision Tree ---
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

# --- K-Means Clustering ---
else:
    st.subheader("K-Means Clustering: Clinical Risk Subtypes")

    kmeans_features = ['Age', 'Heart rate', 'Systolic blood pressure',
                       'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']
    input_kmeans = np.array([[age, heart_rate, sbp, dbp, blood_sugar, ckmb, troponin_for_kmeans]])
    input_scaled_kmeans = scaler_kmeans.transform(input_kmeans)

    cluster_id = kmeans.predict(input_scaled_kmeans)[0]
    st.write(f"🧬 **Assigned Cluster:** {cluster_id}")

    risk_prob = cluster_outcomes.loc[cluster_id, 'HeartAttackRate']
    if risk_prob > 0.8:
        risk_label = "🔴 High"
    elif risk_prob > 0.4:
        risk_label = "🟠 Medium"
    else:
        risk_label = "🟢 Low"

    st.write(f"**Risk Level:** {risk_label} ({risk_prob:.0%} in this cluster)")

    st.markdown("### 🔬 Cluster Profile vs. Patient")
    cluster_mean = cluster_profiles.loc[cluster_id, kmeans_features].values
    categories = kmeans_features + [kmeans_features[0]]
    patient_values = np.append(input_kmeans[0], input_kmeans[0][0])
    cluster_values = np.append(cluster_mean, cluster_mean[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=patient_values, theta=categories, fill='toself', name='Patient'))
    fig.add_trace(go.Scatterpolar(r=cluster_values, theta=categories, fill='toself', name='Cluster Avg'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig)

    st.markdown("### 🧭 Cluster Heatmap Overview")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(cluster_profiles[kmeans_features], annot=True, cmap="coolwarm", fmt=".1f", ax=ax)
    ax.set_title("Average Clinical Features by Cluster")
    st.pyplot(fig)

    st.markdown("### 🧠 Clinical Interpretation")
    if risk_label == "🔴 High":
        st.warning("⚠️ High-risk cluster: features like elevated Troponin or CK-MB likely contributed. Immediate follow-up advised.")
    elif risk_label == "🟠 Medium":
        st.info("🩺 Medium-risk cluster. Patient shows moderate indicators; further testing recommended.")
    else:
        st.success("✅ Low-risk cluster. Continue monitoring and encourage healthy habits.")

        st.markdown("### 📋 Cluster Risk Overview")

    # Static cross-tabulation (from training, not recomputed here)
    cluster_counts = pd.DataFrame({
        'Cluster': [0, 1, 2, 3, 4],
        'No Heart Attack': [153, 183, 22, 0, 88],
        'Heart Attack': [63, 89, 77, 65, 48]
    }).set_index('Cluster')

    cluster_percentages = cluster_counts.div(cluster_counts.sum(axis=1), axis=0).round(2)
    cluster_percentages.columns = ['% No Heart Attack', '% Heart Attack']

    st.markdown("#### 🧮 Raw Cluster Counts")
    st.dataframe(cluster_counts)

    st.markdown("#### 📊 Risk Proportions per Cluster")
    st.dataframe(cluster_percentages)

    # Highlight current user's cluster
    cluster_stats = cluster_percentages.loc[cluster_id]
    st.markdown("#### 🧠 Your Cluster in Context")
    st.write(f"- **You are in Cluster {cluster_id}**.")
    st.write(f"- In this cluster, **{cluster_stats['% Heart Attack']:.0%}** of patients had a heart attack.")

