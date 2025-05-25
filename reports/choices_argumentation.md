# Model Selection Rationale

To predict heart attack risk based on clinical data, I used a combination of supervised and unsupervised models. Each was chosen for its specific strengths in interpretability, performance, or exploration.

## Logistic Regression

- **Why**: Simple, interpretable, and clinically relevant. Coefficients provide direct insight into feature importance.
- **Results**: ~73% accuracy (optimal threshold = 0.40). Top predictors: **Troponin**, **CK-MB**, and **Age**.
- **Usefulness**: Validated key hypotheses and offered a solid, explainable baseline.

## Decision Tree

- **Why**: Handles non-linear patterns well and produces clear decision rules.
- **Results**: ~98% accuracy. Key features: **Troponin**, **CK-MB**, **Glucose**.
- **Usefulness**: Delivered strong performance with human-readable logic. Blood pressure and age were not used in final splits, which offered interesting insights.

## K-Means Clustering

- **Why**: Useful for discovering hidden structure and patient subgroups.
- **Results**: Identified 5 clusters. High-risk groups had elevated **Troponin**, **CK-MB**, and **Glucose**. One high-BP cluster had low risk, challenging assumptions.
- **Usefulness**: Added exploratory value and helped interpret the dataset beyond labels.

## Summary

This model mix allowed me to balance **accuracy**, **interpretability**, and **exploration**. Logistic Regression and Decision Tree supported prediction and explainability, while K-Means helped uncover deeper patterns in the data.
