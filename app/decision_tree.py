import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import _tree

# Load decision tree model
model = joblib.load("outputs/models/decision_tree_model.pkl")

# Preload classification report data
dt_report_train = pd.read_csv("outputs/graphs/dt_classification_report_train.csv", index_col=0)
dt_report_test = pd.read_csv("outputs/graphs/dt_classification_report_test.csv", index_col=0)

def display_results():
    """Interactively traverse the decision tree and display insights."""
    st.subheader("Step-by-Step Decision Tree Walkthrough")
    tree_ = model.tree_
    # Feature names (including one unused 'Gender' for completeness)
    all_features = ["Age", "Gender", "Heart rate", "Systolic blood pressure",
                    "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin"]
    feature_names_used = [all_features[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    # Interactive traversal
    node_id = 0
    step = 0
    show_more = True
    while tree_.children_left[node_id] != _tree.TREE_LEAF and show_more:
        feature = feature_names_used[node_id]
        threshold = tree_.threshold[node_id]
        choice = st.sidebar.radio(
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
    # If reached a leaf node, display the final prediction
    if tree_.children_left[node_id] == _tree.TREE_LEAF:
        value = tree_.value[node_id][0]
        prediction = int(np.argmax(value))
        total = int(np.sum(value))
        probability = value[prediction] / total
        st.markdown("---")
        st.markdown("### Final Decision")
        st.write("**Prediction:**", "🔴 Heart Attack" if prediction == 1 else "🟢 No Heart Attack")
        st.write(f"**Confidence:** {probability:.2%} based on {total} samples")
    # Visual evaluation of the Decision Tree
    st.markdown("## Visual Insights from Decision Tree Model")
    # 1️⃣ Full Tree Diagram
    st.markdown("### 1️⃣ Full Decision Tree Diagram")
    st.image("outputs/graphs/tree_graph.png", caption="Visualized Decision Tree")
    st.markdown(""" 
    - The tree shows how the model splits based on features like **Troponin**, **CK-MB**, and **Blood Sugar**.
    - Each node represents a decision based on a threshold; terminal nodes show predicted class.
    - This helps clinicians understand **why** a patient is classified as high or low risk.
    """)
    # 2️⃣ Confusion Matrix on Test Data
    st.markdown("### 2️⃣ Confusion Matrix on Test Data")
    st.image("outputs/graphs/confusion_matrix_dt.png", caption="Decision Tree Confusion Matrix")
    st.markdown(""" 
    - Diagonal cells represent correct predictions; off-diagonal are errors.
    - The model achieved **high precision and recall**, with minimal misclassifications.
    - Balanced prediction performance across both heart attack and non-heart attack cases.
    """)
    # 3️⃣ Conclusion (model summary)
    st.markdown("### Decision Tree: Summary & Conclusion")
    st.markdown(""" 
    - The **Decision Tree** model provides accurate and interpretable predictions.
    - **Troponin and CK-MB** were key split features, confirming their diagnostic importance.
    - Achieved strong classification results **without requiring ensemble methods**.

    This model offers a clear, logic-based decision pathway suitable for clinical use.
    """)
    # 3️⃣ Classification Report (Training Set)
    st.markdown("### 3️⃣ Classification Report (Training Set)")
    st.dataframe(dt_report_train)
    st.markdown(""" 
    - Displays performance on the **training set**.
    - Precision, recall, and F1-score are all near 1.00 — showing excellent learning.
    - Confirms the Decision Tree captured strong patterns from training data.
    """)
    # 4️⃣ Classification Report (Test Set)
    st.markdown("### 4️⃣ Classification Report (Test Set)")
    st.dataframe(dt_report_test)
    st.markdown(""" 
    - This report evaluates the model on **unseen data** (test set).
    - High precision, recall, and F1-score across both classes show strong generalization.
    - Indicates the Decision Tree performs well in real-world prediction scenarios.
    """)
    # Summary & Hypothesis Check
    st.markdown("## Decision Tree: Summary & Hypothesis Check")
    st.markdown(""" 
    ### Key Takeaways

    - The Decision Tree achieved **~98% accuracy** with clear, interpretable logic paths.
    - **Troponin**, **CK-MB**, and **Blood Sugar** were the most influential features.

    ---

    ### Hypothesis Review

    - **H1 (Troponin & CK-MB):** ✅ Confirmed - key decision splits in the tree.
    - **H2 (BP & Glucose):** ❌ Not supported - while Blood Sugar appears in the tree, its split does not direct toward heart attack predictions; Blood Pressure is not used at all.
    - **H3 (Age):** ❌ Not supported — Age was not used in the model's decision logic.

    ---

    ### Conclusion

    The Decision Tree is an accurate and transparent model for heart attack prediction. It validates key clinical indicators and offers interpretable decision paths for practical use.
    """)
