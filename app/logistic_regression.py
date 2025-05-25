import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load pre-trained scaler and model for logistic regression
scaler = joblib.load("outputs/models/scaler.pkl")
model = joblib.load("outputs/models/logistic_model_final.pkl")

# Feature names for the logistic regression model
logistic_features = [
    "Age", "Heart rate", "Systolic blood pressure", "Diastolic blood pressure",
    "Blood sugar", "CK-MB", "Troponin", "pulse_pressure"
]

def display_results(age, heart_rate, sbp, dbp, blood_sugar, ckmb, troponin, pulse_pressure):
    """Display the logistic regression prediction results and insights."""
    # Prepare input features and scale them
    input_data = np.array([[age, heart_rate, sbp, dbp, blood_sugar, ckmb, troponin, pulse_pressure]])
    input_scaled = scaler.transform(input_data)
    # Predict probability of heart attack
    proba = model.predict_proba(input_scaled)[0][1]
    # Determine risk level based on probability
    if proba >= 0.70:
        risk_label = "🔴 High"
    elif proba >= 0.40:
        risk_label = "🟠 Medium"
    else:
        risk_label = "🟢 Low"
    # Display prediction results
    st.subheader(f"Prediction Result (Logistic Regression)")
    st.write("**Heart Attack Risk Level:**", risk_label)
    st.write(f"**Estimated Probability of Heart Attack:** {proba:.2%}")
    st.markdown(""" 
    This score reflects the model’s prediction based on the entered clinical values. 
    The probability represents how likely it is that the patient has experienced or will experience a heart attack, 
    **according to the logistic regression model** trained on past medical data.
    """)
    # Gauge chart for risk score (%)
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
    # Clinical interpretation message
    st.markdown("**Clinical Interpretation**")
    if proba >= 0.70:
        st.warning("⚠️ This patient is at **high risk**. Recommend urgent follow-up.")
    elif proba >= 0.40:
        st.info("🩺 Medium risk. Suggest follow-up tests or monitoring.")
    else:
        st.success("✅ Low risk. Encourage healthy lifestyle and regular checkups.")
    # Calculate feature influence (coefficient * scaled value)
    coef_dict = dict(zip(logistic_features, model.coef_[0]))
    scaled_vals = input_scaled[0]
    influence = {feat: coef_dict[feat] * scaled_vals[i] for i, feat in enumerate(logistic_features)}
    # Show top feature contributors
    st.markdown("### Top Feature Contributors")
    for feat, val in sorted(influence.items(), key=lambda x: abs(x[1]), reverse=True):
        sign = "↑" if val > 0 else "↓"
        st.write(f"- **{feat}**: {sign} contributed **{val:.2f}**")
    # Coefficient table
    st.markdown("## Logistic Regression Coefficients")
    coef_series = pd.Series(model.coef_[0], index=logistic_features).sort_values(ascending=False)
    st.dataframe(coef_series.rename("Coefficient").to_frame())
    st.markdown(
        "- Coefficients represent the **log-odds impact** of each feature.\\n"
        "- **Positive values** indicate higher risk when the feature increases (e.g., Troponin).\\n"
        "- **Negative values** suggest a protective or inverse association."
    )
    st.markdown("**Observation**: Troponin dominates the model with the highest positive influence.")
    # Visual insights from training phase
    st.markdown("## Visual Insights from Training Phase")
    # 1️⃣ Troponin Levels by Heart Attack Result
    st.markdown("### 1️⃣ Troponin Levels by Heart Attack Result")
    st.image("outputs/graphs/troponin_levels_by_heart_attack.png", caption="Boxplot of Troponin Levels")
    st.markdown(
        "- Troponin is significantly higher in patients with heart attacks.\\n"
        "- Confirms why Troponin received the **strongest model weight**.\\n"
        "- High clinical diagnostic value in distinguishing between outcome classes."
    )
    # 2️⃣ Confusion Matrix on Test Data
    st.markdown("### 2️⃣ Confusion Matrix on Test Data")
    st.image("outputs/graphs/confusion_matrix_lr.png", caption="Confusion Matrix (Default Threshold)")
    st.markdown(
        "- Diagonal values show correct predictions; off-diagonal are errors.\\n"
        "- The model achieved **70% accuracy** with a balanced error spread.\\n"
        "- Helps identify trade-offs between false positives and false negatives."
    )
    # 3️⃣ Cross-Validation Performance (Fold-wise)
    st.markdown("### 3️⃣ Cross-Validation Performance (Fold-wise)")
    st.image("outputs/graphs/cv_performance.png", caption="5-Fold CV Performance")
    st.markdown(
        "- Shows model stability across 5 data splits.\\n"
        "- Performance metrics like precision, recall, F1 score are consistent.\\n"
        "- Helps justify Fold 3 as the best candidate for final model training."
    )
    # 4️⃣ Threshold Tuning (Fold 3 Final Model)
    st.markdown("### 4️⃣ Threshold Tuning (Fold 3 Final Model)")
    st.image("outputs/graphs/threshold_tuning.png", caption="Threshold Optimization")
    st.markdown(
        "- Evaluated performance at different probability cutoffs.\\n"
        "- Threshold **0.40** offered the **best F1 score** and clinical safety.\\n"
        "- Used this threshold to label high/medium/low risk categories in predictions."
    )
    # 5️⃣ Threshold Tuning Results
    st.markdown("### 5️⃣ Threshold Tuning Results")
    threshold_results = pd.read_csv("outputs/models/threshold_tuning_summary.csv")
    st.dataframe(threshold_results.round(3))
    st.markdown(""" 
        - Each row represents model performance at a specific classification threshold (probability cutoff).
        - **Lower thresholds (e.g., 0.10–0.25)** increase sensitivity (**recall**) but reduce specificity (**precision**), leading to more false positives.
        - **Higher thresholds (e.g., 0.50–0.55)** increase precision but miss more actual heart attack cases (lower recall).
        - The **optimal threshold is 0.40**, achieving the best F1 score (**0.719**), strong recall (**0.797**), and high accuracy (**72.8%**).

        This threshold was selected for use in the app to balance **clinical safety** and **prediction accuracy**.
    """)
    # Summary & Hypothesis Check
    st.markdown("## Logistic Regression: Summary & Hypothesis Check")
    st.markdown(""" 
    ### Key Takeaways

    - Logistic Regression provided a strong, interpretable model with ~73% accuracy and optimal performance at a **0.40 threshold**.
    - **Troponin** emerged as the most influential predictor, followed by **CK-MB**, aligning with clinical expectations.

    ---

    ### Hypothesis Review

    - **H1 (Troponin & CK-MB):** ✅ Confirmed - strong positive coefficients and clinical relevance.
    - **H2 (BP & Glucose):** ❌ Not supported - coefficients suggest weak or inverse relationships.
    - **H3 (Age):** ✅ Supported - age showed positive correlation, though modest in magnitude.

    ---

    ### Conclusion

    Logistic regression validated key biomarkers and offered actionable insights for early heart attack detection. It serves well as a baseline for risk stratification and supports future integration of broader clinical features.
    """)
