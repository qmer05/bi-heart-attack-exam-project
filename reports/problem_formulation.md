# Project Title: Tech Life Balance – A BI-Driven Approach to Mental Wellbeing

## 1. Project Annotation

In this Business Intelligence (BI) project, I investigate the relationship between digital behavior patterns—such as screen time, device usage, and online activity—and their impact on mental health indicators like stress, mood, sleep quality, and anxiety. My dataset includes user-reported data on device usage, daily habits, and wellness metrics. The aim is to identify hidden patterns and correlations that can guide users toward healthier digital lifestyles.

This challenge is highly relevant as digital overuse has become a common yet often overlooked factor affecting mental health. The expected outcome of the project is an interactive BI dashboard and report offering insights, predictive trends, and practical recommendations based on user profiles. The solution could positively impact individuals, mental wellness app providers, and institutional wellness programs (e.g., universities, HR departments).

---

## 2. Problem Statement

### Context
Digital devices are now central to everyday life, but prolonged use—especially on phones and social media—has been associated with decreased mental health. By analyzing behavioral and mental health data, I aim to better understand how screen time and lifestyle choices affect well-being.

### Purpose
To use BI tools to analyze behavioral patterns and provide actionable insights that promote healthier digital habits and better mental health outcomes.

### Research Questions
- How do various forms of screen time correlate with stress levels, mood ratings, and sleep quality?
- What combinations of digital usage and lifestyle factors lead to higher or lower mental health scores?
- Can we segment users into risk groups based on their habits?
- What recommendations can be derived for different user profiles?

### Hypotheses
- H1: Higher daily phone usage is positively correlated with stress and lower mood ratings.
- H2: Users with more mindfulness minutes and lower caffeine intake show better sleep quality and mental health scores.
- H3: Individuals living in urban areas have higher screen time and stress levels than those in suburban locations.

---

## 3. Development Plan

### Timeline & Milestones
| Sprint | Duration       | Goals                                                | Deliverables                         |
|--------|----------------|------------------------------------------------------|--------------------------------------|
| 1      | Week 1         | Problem formulation, planning                        | Problem statement, .md file          |
| 2      | Weeks 2–3      | Data cleaning, exploration, and correlation analysis | Data report, initial visualizations  |
| 3      | Weeks 4–5      | BI dashboard and optional ML model                   | Interactive dashboard, user insights |
| 4      | Week 6         | Final report and GitHub documentation                | Final .pdf, GitHub release           |

### Tools and Environment
- **Programming**: Python (Pandas, Plotly, Matplotlib, Scikit-learn)
- **Visualization**: Streamlit for web app
- **Version Control**: Git & GitHub
- **Documentation**: Markdown (.md), Jupyter Notebooks

### Repository Structure

/bi-tech-life-balance
│
├── data/                         # Contains your primary dataset and any cleaned versions
│   └── user_behavior_mental_health.csv
│
├── notebooks/                   # Contains analysis and modeling notebooks
│   ├── 01_data_cleaning.ipynb             # Sprint 2
│   ├── 02_exploratory_analysis.ipynb      # Sprint 2
│   ├── 03_modeling.ipynb                  # Sprint 3
│   └── 04_visualization_interface.ipynb   # Sprint 4
│
├── reports/                     # Project documentation and PDF exports
│   ├── sprint1_problem_formulation.md
│   ├── sprint2_data_preparation.md
│   ├── sprint3_modeling_summary.md
│   ├── sprint4_visualisation_summary.md
│   └── final_report.pdf
│
├── app/                         # Contains final deployment (optional web app or Streamlit interface)
│   ├── main.py
│   └── assets/
│
├── dashboards/                  # Optional BI dashboards (Power BI, Tableau)
│   └── dashboard.pbix
│
├── outputs/                     # Exported charts, models, datasets
│   ├── figures/
│   └── models/
│
├── README.md                    # Overview of project and instructions
├── requirements.txt             # List of Python libraries used
└── .gitignore                   # Files to ignore in version control



---

## 4. Team Structure (if applicable)

- Data Analyst: Responsible for cleaning and exploring the data
- BI Developer: Builds dashboards and visualizations
- ML Engineer: Develops any predictive models
- Project Lead: Coordinates planning, documentation, and delivery

