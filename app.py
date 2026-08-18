"""
Streamlit App: Customer Segmentation Dashboard
------------------------------------------------
Run locally with:  streamlit run app.py
Deploy free at:     https://share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", layout="wide", page_icon="🛍️")
sns.set_style("whitegrid")

st.title("🛍️ Customer Segmentation using K-Means Clustering")
st.caption("Python · Scikit-learn · Pandas · Seaborn · Streamlit")

# ---------------------------------------------------------
# SIDEBAR — DATA + PARAMETERS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Settings")

uploaded = st.sidebar.file_uploader("Upload your own CSV (optional)", type=["csv"])

@st.cache_data
def load_default_data():
    return pd.read_csv("data/Mall_Customers.csv")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Custom dataset loaded ✅")
else:
    df = load_default_data()
    st.sidebar.info("Using sample dataset (Mall_Customers.csv)")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# drop obvious ID-like columns from the default feature suggestions
default_feats = [c for c in numeric_cols if "id" not in c.lower()]

features = st.sidebar.multiselect(
    "Features to cluster on",
    options=numeric_cols,
    default=default_feats[:3] if len(default_feats) >= 3 else default_feats
)

k = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=10, value=5)

if len(features) < 2:
    st.warning("Please select at least 2 numeric features from the sidebar to run clustering.")
    st.stop()

# ---------------------------------------------------------
# CLUSTERING
# ---------------------------------------------------------
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_result = df.loc[X.index].copy()
df_result["Cluster"] = clusters
sil = silhouette_score(X_scaled, clusters)

# ---------------------------------------------------------
# TOP METRICS
# ---------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Customers analyzed", len(df_result))
c2.metric("Clusters (K)", k)
c3.metric("Silhouette score", f"{sil:.3f}")

st.divider()

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🎯 Clusters", "🧾 Segment Profiles", "📥 Data"])

with tab1:
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        feat_for_hist = st.selectbox("Feature to view distribution", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[feat_for_hist], kde=True, ax=ax, color="steelblue")
        st.pyplot(fig)
    with col2:
        st.write("Summary statistics")
        st.dataframe(df[numeric_cols].describe().round(2))

    if len(features) >= 2:
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x=features[0], y=features[1], ax=ax2)
        ax2.set_title(f"{features[0]} vs {features[1]} (before clustering)")
        st.pyplot(fig2)

with tab2:
    st.subheader("Customer Segments")
    if len(features) >= 2:
        x_axis = st.selectbox("X-axis", features, index=0, key="x_axis")
        y_axis = st.selectbox("Y-axis", features, index=1, key="y_axis")

        fig3, ax3 = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            data=df_result, x=x_axis, y=y_axis, hue="Cluster",
            palette="tab10", s=70, ax=ax3
        )
        ax3.set_title(f"K-Means Segments (K={k})")
        st.pyplot(fig3)

    if len(features) >= 3:
        st.write("Pairwise relationships across all selected features")
        pair_fig = sns.pairplot(df_result, vars=features, hue="Cluster", palette="tab10")
        st.pyplot(pair_fig)

with tab3:
    st.subheader("Segment Profiles & Business Insights")
    profile = df_result.groupby("Cluster")[features].mean().round(1)
    profile["Count"] = df_result["Cluster"].value_counts().sort_index()
    profile["% of customers"] = (profile["Count"] / profile["Count"].sum() * 100).round(1)
    st.dataframe(profile)

    st.markdown("#### 💡 Suggested targeting per segment")
    for cluster_id, row in profile.iterrows():
        st.markdown(f"**Cluster {cluster_id}** — {int(row['Count'])} customers ({row['% of customers']}%)")
        summary = ", ".join([f"{f}: {row[f]}" for f in features])
        st.caption(summary)
    st.info(
        "Use these profiles to tailor marketing: high spenders with lower income may respond to "
        "loyalty/discount programs, high-income low-spenders may need premium/aspirational campaigns, "
        "and high-income high-spenders are ideal for premium retention and upsell offers."
    )

with tab4:
    st.subheader("Segmented Dataset")
    st.dataframe(df_result)
    csv = df_result.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download segmented data as CSV", csv, "segmented_customers.csv", "text/csv")

st.divider()
st.caption("Built with K-Means clustering · Adjust features and K in the sidebar to explore different segmentations.")
