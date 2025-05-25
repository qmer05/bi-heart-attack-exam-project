# data_exploration.py
import streamlit as st
import pandas as pd

def show_exploration_page():
    """Display the data exploration and analysis content on the Streamlit app."""
    # Page title and introduction
    st.title("Data Exploration and Analysis")
    st.markdown("""
    This section provides an overview of the dataset and key insights from exploratory data analysis.
    Use this page to understand the data **before** exploring predictive models.
    """)

    # 1. Dataset preview (first 5 rows)
    st.subheader("Dataset Preview")
    df = pd.read_csv('data/cleaned_Medicaldataset.csv')
    st.dataframe(df.head())  # show first five records
    st.markdown("*Above are the first five entries of the dataset, showing all features and the heart attack outcome.*")

    # 1a. Dataset summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe().transpose())
    st.markdown("*This table shows the mean, standard deviation, min, max, and quartile values for each numerical feature. It helps understand the scale and variability of the features before modeling.*")

    # 2. Feature Distributions (Histograms for each feature)
    st.subheader("Feature Distributions")
    st.markdown("Each histogram below shows the distribution of a clinical feature across all patients:")
    # Age distribution
    st.image("outputs/graphs/hist_age.png", caption="Age Distribution", use_container_width=True)
    st.markdown("- **Age:** The distribution is roughly bell-shaped with a slight right skew. Most patients are between 45 and 70 years old, peaking around 60–65. There are fewer very young and very elderly patients in the dataset.")
    # Heart rate distribution
    st.image("outputs/graphs/hist_heart_rate.png", caption="Heart Rate Distribution", use_container_width=True)
    st.markdown("- **Heart Rate:** Right-skewed distribution with a peak around 60–65 bpm. Most patients have heart rates between 60 and 90 bpm, with fewer cases extending above 100 bpm, indicating occasional elevated heart rates.")
    # Systolic BP distribution
    st.image("outputs/graphs/hist_systolic_blood_pressure.png", caption="Systolic BP Distribution", use_container_width=True)
    st.markdown("- **Systolic Blood Pressure:** The distribution is close to normal with a peak around 115–125 mmHg. Most patients have systolic pressures between 100 and 150 mmHg. A small number of hypertensive outliers exceed 170 mmHg.")
    # Diastolic BP distribution
    st.image("outputs/graphs/hist_diastolic_blood_pressure.png", caption="Diastolic BP Distribution", use_container_width=True)
    st.markdown("- **Diastolic Blood Pressure:** Distribution is centered around 75–80 mmHg, with most patients falling between 60 and 90 mmHg. The shape is fairly symmetric, with a few rare outliers on both ends.")
    # Blood sugar distribution
    st.image("outputs/graphs/hist_blood_sugar.png", caption="Blood Sugar Distribution", use_container_width=True)
    st.markdown("- **Blood Sugar:** Strongly right-skewed distribution, peaking around 90–110 mg/dL. Most patients fall within the normal range, but a long tail of elevated values extends beyond 250 mg/dL, suggesting a subset with hyperglycemia or diabetes.")
    # CK-MB distribution
    st.image("outputs/graphs/hist_ck_mb.png", caption="CK-MB Level Distribution", use_container_width=True)
    st.markdown("- **CK-MB:** Strongly right-skewed distribution with a sharp peak around 1–2 ng/mL. Most patients have low CK-MB levels, but a noticeable subgroup shows elevated values above 5 ng/mL, consistent with heart muscle injury or acute coronary events.")
    # Troponin distribution
    st.image("outputs/graphs/hist_troponin.png", caption="Troponin Level Distribution", use_container_width=True)
    st.markdown("- **Troponin:** Extremely right-skewed distribution with a sharp peak near 0 ng/mL. Most patients have baseline troponin levels, but a distinct minority exhibit markedly elevated values—clear indicators of acute heart muscle damage consistent with heart attacks.")

    # 3. Outcome (Heart Attack) distribution
    st.subheader("Heart Attack Outcome Distribution")
    st.image("outputs/graphs/result_distribution.png", caption="Heart Attack vs No Heart Attack", use_container_width=True)
    # Calculate counts to display textually (optional)
    outcome_counts = df[df.columns[-1]].value_counts()  # assume last column is outcome
    no_count = int(outcome_counts.get(0, 0))
    yes_count = int(outcome_counts.get(1, 0))
    total = no_count + yes_count
    st.markdown(f"- **No Heart Attack:** {no_count} patients  \n- **Heart Attack:** {yes_count} patients")
    st.markdown(f"Out of **{total} patients**, about **{no_count/total:.0%}** had no heart attack and **{yes_count/total:.0%}** had a heart attack. This slight class imbalance will be considered in model training.")

    # 4. Correlation Matrix
    st.subheader("Correlation Matrix")
    st.image("outputs/graphs/correlation_heatmap.png", caption="Feature Correlation Heatmap", use_container_width=True)
    st.markdown("""The correlation matrix above shows how each feature relates to others and to the outcome:
- **Troponin** shows a strong positive correlation with the heart attack outcome (**0.54**), making it a key predictor.
- **CK-MB** also correlates moderately with the outcome (**0.29**), suggesting diagnostic value but to a lesser extent than troponin.
- **Systolic and Diastolic Blood Pressure** are highly correlated (**0.60**), as expected due to their physiological relationship.
- **Troponin and CK-MB**, while both related to cardiac injury, are **not strongly correlated** in this dataset.
- Other features (like age, heart rate, blood sugar) show **weak correlations** with the outcome, suggesting limited predictive power on their own.
""")


    # 5. Average Biomarker Levels by Outcome
    st.subheader("Average Biomarker Levels by Outcome")
    col1, col2 = st.columns(2)
    # Average CK-MB bar chart
    with col1:
        st.image("outputs/graphs/bar_mean_ckmb_by_result.png", caption="Average CK-MB by Outcome", use_container_width=True)
        st.markdown("""**CK-MB:** Patients who had heart attacks show a much higher average CK-MB level compared to those who did not. This aligns with CK-MB being a marker released during heart muscle injury (significantly elevated in heart attack cases).""")
    # Average Troponin bar chart
    with col2:
        st.image("outputs/graphs/bar_mean_troponin_by_result.png", caption="Average Troponin by Outcome", use_container_width=True)
        st.markdown("""**Troponin:** Similarly, the average troponin level for heart attack patients is dramatically higher than for non-heart-attack patients. Troponin is nearly normal for the no-heart-attack group, but **elevated by several-fold** in the heart attack group, underscoring its diagnostic importance.""")    