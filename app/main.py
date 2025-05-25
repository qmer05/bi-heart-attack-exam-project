import streamlit as st
# Import model modules
import logistic_regression
import decision_tree
import kmeans_clustering

# Title and description
st.title("Heart Attack Risk Prediction Dashboard")
st.markdown("A diagnostic support tool powered by Machine Learning and BI insights.")
st.sidebar.header("Enter Patient Data")

# Sidebar model selection
model_option = st.sidebar.selectbox(
    "Choose Prediction Model",
    ["Logistic Regression", "Decision Tree", "K-Means Clustering"]
)

# Shared input sliders for Logistic Regression and K-Means
if model_option != "Decision Tree":
    age = st.sidebar.slider("Age", 18, 99, 56, format="%d years")
    heart_rate = st.sidebar.slider("Heart Rate", 35, 130, 75, format="%d bpm")
    sbp = st.sidebar.slider("Systolic Blood Pressure", 60, 200, 130, format="%d mmHg")
    dbp = st.sidebar.slider("Diastolic Blood Pressure", 35, 120, 70, format="%d mmHg")
    blood_sugar = st.sidebar.slider("Blood Sugar", 30, 280, 120, format="%d mg/dL")
    ckmb = st.sidebar.slider("CK-MB", 0.30, 10.99, 2.49, format="%.2f ng/mL")
    troponin_ui = st.sidebar.slider("Troponin", 1, 30, 5, format="%d ng/mL")
    # Convert troponin units for different models
    troponin_for_logistic = troponin_ui / 1000.0
    troponin_for_kmeans = troponin_ui / 100.0
    # Additional feature for logistic regression
    pulse_pressure = sbp - dbp

# Render the appropriate model section
if model_option == "Logistic Regression":
    logistic_regression.display_results(
        age, heart_rate, sbp, dbp, blood_sugar, ckmb, troponin_for_logistic, pulse_pressure
    )
elif model_option == "Decision Tree":
    decision_tree.display_results()
elif model_option == "K-Means Clustering":
    kmeans_clustering.display_results(
        age, heart_rate, sbp, dbp, blood_sugar, ckmb, troponin_for_kmeans
    )
