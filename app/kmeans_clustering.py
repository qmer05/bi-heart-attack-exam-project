import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load K-Means model, scaler, and cluster data
scaler_kmeans = joblib.load("outputs/models/scaler_kmeans.pkl")
kmeans = joblib.load("outputs/models/kmeans_model.pkl")
cluster_profiles = pd.read_csv("outputs/models/cluster_feature_means.csv")
cluster_outcomes = pd.read_csv("outputs/models/cluster_outcome_ratios.csv")

# Predefined cluster counts from training data
cluster_counts = pd.DataFrame({
    'Cluster': [0, 1, 2, 3, 4],
    'No Heart Attack': [153, 183, 22, 0, 88],
    'Heart Attack': [63, 89, 77, 65, 48]
}).set_index('Cluster')
cluster_percentages = cluster_counts.div(cluster_counts.sum(axis=1), axis=0).round(2)
cluster_percentages.columns = ['% No Heart Attack', '% Heart Attack']

def display_results(age, heart_rate, sbp, dbp, blood_sugar, ckmb, troponin):
    """Display clustering results and insights."""
    st.subheader("K-Means Clustering: Clinical Risk Subtypes")
    # Features used for K-Means clustering
    kmeans_features = ['Age', 'Heart rate', 'Systolic blood pressure',
                       'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']
    # Scale input for clustering
    input_data = np.array([[age, heart_rate, sbp, dbp, blood_sugar, ckmb, troponin]])
    input_scaled = scaler_kmeans.transform(input_data)
    # Predict cluster assignment
    cluster_id = kmeans.predict(input_scaled)[0]
    st.write(f" **Assigned Cluster:** {cluster_id}")
    # Determine risk level from cluster outcomes
    risk_prob = cluster_outcomes.loc[cluster_id, 'HeartAttackRate']
    if risk_prob > 0.8:
        risk_label = "🔴 High"
    elif risk_prob > 0.4:
        risk_label = "🟠 Medium"
    else:
        risk_label = "🟢 Low"
    st.write(f"**Risk Level:** {risk_label} ({risk_prob:.0%} in this cluster)")
    # Radar chart: patient vs cluster average profile
    st.markdown("### Cluster Profile vs. Patient")
    cluster_mean = cluster_profiles.loc[cluster_id, kmeans_features].values
    categories = kmeans_features + [kmeans_features[0]]
    patient_values = np.append(input_data[0], input_data[0][0])
    cluster_values = np.append(cluster_mean, cluster_mean[0])
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=patient_values, theta=categories, fill='toself', name='Patient'))
    fig.add_trace(go.Scatterpolar(r=cluster_values, theta=categories, fill='toself', name='Cluster Avg'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig)
    # Clinical interpretation
    st.markdown("### Clinical Interpretation")
    if risk_label == "🔴 High":
        st.warning("⚠️ High-risk cluster: features like elevated Troponin or CK-MB likely contributed. Immediate follow-up advised.")
    elif risk_label == "🟠 Medium":
        st.info("🩺 Medium-risk cluster. Patient shows moderate indicators; further testing recommended.")
    else:
        st.success("✅ Low-risk cluster. Continue monitoring and encourage healthy habits.")
        st.markdown("### Cluster Risk Overview")
    # Cluster outcome distribution and context
    st.markdown("#### Raw Cluster Counts")
    st.dataframe(cluster_counts)
    st.markdown("#### Risk Proportions per Cluster")
    st.dataframe(cluster_percentages)
    # Highlight current user's cluster statistics
    cluster_stats = cluster_percentages.loc[cluster_id]
    st.markdown("#### Your Cluster in Context")
    st.write(f"- **You are in Cluster {cluster_id}**.")
    st.write(f"- In this cluster, **{cluster_stats['% Heart Attack']:.0%}** of patients had a heart attack.")
    # Visual insights from clustering
    st.markdown("## Visual Insights from K-Means Clustering")
    # 1️⃣ Elbow Method for optimal k
    st.markdown("### 1️⃣ Elbow Method: Optimal Cluster Count")
    st.image("outputs/graphs/elbow_method_optimal_k.png", caption="Elbow Method Curve")
    st.markdown(
        "- The **elbow point around k = 5** shows where adding more clusters stops significantly reducing distortion.\\n"
        "- This supports using **5 clusters** as a meaningful balance between accuracy and simplicity."
    )
    # 2️⃣ Silhouette Scores by k
    st.markdown("### 2️⃣ Silhouette Scores for Different k")
    st.image("outputs/graphs/silhouette_method_optimal_k.png", caption="Silhouette Score by k")
    st.markdown(
        "- **Silhouette score peaks at k = 5**, confirming it as the best-separated configuration.\\n"
        "- A higher silhouette score reflects better-defined and well-separated clusters."
    )
    # 3️⃣ Silhouette Plot for k = 5
    st.markdown("### 3️⃣ Silhouette Plot for k = 5")
    st.image("outputs/graphs/silhouette_plot_kmeans_clustering.png", caption="Silhouette Plot for KMeans")
    st.markdown(
        "- Each bar shows how well a sample fits within its cluster.\\n"
        "- Most samples have positive silhouette values, confirming that **no major overlaps or weak clusters** exist."
    )
    # 4️⃣ Cluster Visualization (PCA)
    st.markdown("### 4️⃣ Cluster Visualization (PCA)")
    st.image("outputs/graphs/tsne_visualization_clusters.png", caption="t-SNE Cluster Projection")
    st.markdown(
        "- This 2D view shows **well-separated clusters** with distinct spatial groupings.\\n"
        "- It confirms that the KMeans model effectively differentiated patient subtypes."
    )
    # 5️⃣ Cluster Boundaries with t-SNE
    # (No heading output for 5; image and description only)
    st.image("outputs/graphs/tsne_cluster_visualization_with_kmeans_boundaries.png", caption="t-SNE Boundaries")
    st.markdown(
        "- Decision boundaries visually outline where one cluster ends and another begins.\\n"
        "- **Black stars show centroids**, helping interpret cluster centers and proximity."
    )
    # 6️⃣ Clusters vs. Heart Attack Outcomes
    st.markdown("### 6️⃣ Clusters vs. Heart Attack Outcomes")
    st.image("outputs/graphs/tsne_clusters_vs_heart_attacks.png", caption="t-SNE Clusters vs Labels")
    st.markdown(
        "- Red X's represent actual heart attack cases.\\n"
        "- High-risk clusters (like **Cluster 3 and 2**) show dense overlap with heart attack labels, validating their clinical relevance."
    )
    # 7️⃣ Cluster Feature Profiles (Standardized)
    st.markdown("### 7️⃣ Cluster Feature Profiles (Standardized)")
    st.image("outputs/graphs/standardized_feature_values_per_cluster.png", caption="Heatmap: Cluster Profiles")
    st.markdown(
        "- Cluster 3 has **very high Troponin**, Cluster 2 shows **elevated CK-MB**, and Cluster 4 has **high Blood Sugar** — all linked to increased risk.\\n"
        "- Cluster 0 has the **highest BP values**, yet the **lowest heart attack rate**, suggesting BP alone isn’t a strong predictor here."
    )
    # Summary & Hypothesis Check
    st.markdown("## K-Means Clustering: Summary & Hypothesis Check")
    st.markdown(""" 
    ### Key Takeaways

    - K-Means identified **5 distinct patient subtypes** with unique clinical profiles.
    - Clusters with **high Troponin, CK-MB, and Glucose** showed higher heart attack rates.
    - Visualizations confirmed that clusters were **well-separated** and meaningful.

    ---

    ### Hypothesis Review

    - **H1 (Troponin & CK-MB):** ✅ Confirmed — linked to high-risk clusters.
    - **H2 (BP & Glucose):** ❌ Not fully supported — although elevated in some clusters, they don't correspond with the highest heart attack risk. In fact, the lowest-risk cluster has the highest BP.
    - **H3 (Age):** ✅ Supported — older clusters had higher event rates.

    ---

    ### Conclusion

    K-Means clustering provided valuable insights into **unsupervised risk stratification** and validated key predictors without using labels — supporting its use in early, exploratory diagnostics.
    """)
