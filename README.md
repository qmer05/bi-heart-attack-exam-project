# Project Title: Predicting Heart Attacks – A BI-Powered Diagnostic Support Tool

## 1. Project Annotation

In this Business Intelligence (BI) project, I analyze a clinical dataset collected from Zheen Hospital in Erbil, Iraq, to explore and model the risk factors associated with heart attacks. The dataset includes medical indicators such as heart rate, blood pressure, blood sugar, and cardiac biomarkers (CK-MB, Troponin), along with demographic attributes like age and gender. The project’s primary objective is to discover data-driven insights and develop an interactive BI dashboard to support early diagnosis and prevention strategies.

Heart attacks remain a leading cause of mortality globally. Early identification of at-risk individuals using clinical data can greatly improve outcomes. The expected deliverables include a dashboard that visualizes health indicators and predictive trends, and a final report highlighting actionable insights for healthcare practitioners, researchers, and institutional health programs.

---

## 2. Problem Statement

### Context
Heart diseases are often preventable, but early detection remains a challenge due to the multifactorial nature of cardiovascular risk. With the growing availability of patient health records, BI and machine learning tools can play a pivotal role in risk stratification and clinical decision-making.

### Purpose
To use BI and predictive modeling techniques to identify correlations among vital signs and biomarkers, and to build a tool that helps in assessing the likelihood of a heart attack based on patient data.

### Research Questions
- What clinical indicators are most strongly associated with a positive heart attack diagnosis?
- Are there significant patterns across demographics (age, gender) that influence heart attack outcomes?
- Can we build a predictive model that classifies patients into risk categories?
- What trends can be visualized to assist medical professionals in preventative diagnostics?

### Hypotheses
- H1: Elevated troponin and CK-MB levels are strong predictors of heart attack occurrence.
- H2: Patients with higher blood pressure and glucose levels show higher likelihood of a heart attack.
- H3: Age is positively correlated with the probability of a heart attack.

---

## 3. Development Plan

### Timeline & Milestones
| Sprint | Duration       | Goals                                                | Deliverables                         |
|--------|----------------|------------------------------------------------------|--------------------------------------|
| 1      | Week 1         | Problem formulation, planning                        | Problem statement, .md file          |
| 2      | Weeks 2–3      | Data cleaning, exploration, and correlation analysis | Data report, initial visualizations  |
| 3      | Weeks 4–5      | BI dashboard and predictive modeling                 | Interactive dashboard, model summary |
| 4      | Week 6         | Final report and GitHub documentation                | Final .pdf, GitHub release           |

### Tools and Environment
- **Programming**: Python (Pandas, Matplotlib, Seaborn, Scikit-learn)
- **Visualization**: Streamlit or Power BI for the dashboard
- **Version Control**: Git & GitHub
- **Documentation**: Markdown (.md), Jupyter Notebooks

### Repository Structure

<pre> ### 📁 Repository Structure 
``` /bi-heart-attack-exam-project ├── app/ # Streamlit application files for deployment │ └── app.py # Main app entry point │ ├── data/ # Raw and cleaned datasets │ ├── Medicaldataset.csv │ └── cleaned_Medicaldataset.csv │ ├── outputs/ # Model outputs and visualizations │ ├── models/ # Trained model files & scalers │ │ ├── logistic_model_final.pkl │ │ ├── decision_tree_model.pkl │ │ ├── kmeans_model.pkl │ │ ├── scaler.pkl │ │ ├── scaler_kmeans.pkl │ │ ├── cluster_feature_means.csv │ │ └── cluster_outcome_ratios.csv │ ├── graphs/ # Visualizations used in the app │ └── threshold_tuning_summary.csv │ ├── reports/ # Final report and hypothesis notes │ ├── Exam-Project.pdf │ └── hypotheses_details.md │ ├── sprint_2_data_preparation/ # Data cleaning and preprocessing notebooks │ ├── 02_data_cleaning.ipynb │ ├── 03_exploration_and_analysis_copy.ipynb │ └── 04_data_engineering.ipynb │ ├── sprint_3_data_modelling/ # Modeling and evaluation notebooks │ ├── decision_tree.ipynb │ ├── k_means_clustering.ipynb │ ├── logistic_regression.ipynb │ └── different_data_models.ipynb │ ├── README.md # Project overview and documentation ├── requirements.txt # Python dependencies └── .gitignore # Files/folders to exclude from version control ``` </pre>

---

## 4. Team Structure (if applicable)

- **Data Analyst**: Cleans data and performs initial statistical analysis
- **BI Developer**: Designs and builds visual dashboards
- **ML Engineer**: Develops and evaluates predictive models
- **Project Lead**: Oversees documentation, coordination, and final delivery