import streamlit as st
# Import model modules
import logistic_regression
import decision_tree
import kmeans_clustering

# --- UI Setup ---
st.title("Heart Attack Risk Prediction Dashboard")
st.markdown("A diagnostic support tool powered by Machine Learning and BI insights.")

# Sidebar: Page/Model selector (now including Data Exploration)
st.sidebar.header("Navigation")
page_option = st.sidebar.selectbox("Choose a page or model:", [
    "Data Exploration",            # new default page
    "Logistic Regression", 
    "Decision Tree", 
    "K-Means Clustering"
])

# Sidebar: Patient data input (Logistic Regression & K-Means pages)
if page_option in ["Logistic Regression", "K-Means Clustering"]:
    st.sidebar.subheader("Enter Patient Data")
    age = st.sidebar.slider("Age", 18, 99, 56, format="%d years")
    heart_rate = st.sidebar.slider("Heart Rate", 35, 130, 75, format="%d bpm")
    sbp = st.sidebar.slider("Systolic Blood Pressure", 60, 200, 130, format="%d mmHg")
    dbp = st.sidebar.slider("Diastolic Blood Pressure", 35, 120, 70, format="%d mmHg")
    blood_sugar = st.sidebar.slider("Blood Sugar", 30, 280, 120, format="%d mg/dL")
    ckmb = st.sidebar.slider("CK-MB", 0.30, 10.99, 2.49, format="%.2f ng/mL")
    # Updated Troponin slider: use µg/mL (0.001–0.030) instead of ng/mL (1–30)
    troponin_ui = st.sidebar.slider(
        "Troponin", 0.001, 0.030, 0.005, step=0.001, format="%.3f µg/mL"
    )
    # Prepare any transformed inputs for models (adjusted for new units)
    troponin_for_logistic = troponin_ui            # already in µg/mL, no further scaling
    troponin_for_kmeans  = troponin_ui * 10        # equivalent to old (ng/mL)/100 scaling
    pulse_pressure = sbp - dbp

# Page routing logic
if page_option == "Data Exploration":

    st.sidebar.info("""
    **Heart Attack Risk Explorer**

    This diagnostic tool leverages clinical data and machine learning to assess the likelihood of a heart attack.

    **Features:**
    - Data exploration and biomarker insights  
    - Risk prediction with multiple ML models  
    - Interactive interface for patient-level analysis
    """)

    # Display the data exploration start page
    from data_exploration import show_exploration_page
    show_exploration_page()

# Render the appropriate model section
elif page_option == "Logistic Regression":
    logistic_regression.display_results(
        age, heart_rate, sbp, dbp, blood_sugar, ckmb, troponin_for_logistic, pulse_pressure
    )
elif page_option == "Decision Tree":
    decision_tree.display_results()
elif page_option == "K-Means Clustering":
    kmeans_clustering.display_results(
        age, heart_rate, sbp, dbp, blood_sugar, ckmb, troponin_for_kmeans
    )