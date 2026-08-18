"""
Customer Segmentation using K-Means Clustering
------------------------------------------------
Tools: Python, Pandas, Scikit-learn, Seaborn

Pipeline:
1. Load & explore data (EDA)
2. Preprocess / scale features
3. Find optimal K (Elbow Method + Silhouette Score)
4. Fit K-Means
5. Profile & visualize clusters
6. Save model + labeled dataset for reuse (e.g. in the Streamlit app)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os

sns.set_style("whitegrid")
os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("data/Mall_Customers.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# ---------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["Age"], kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Age Distribution")
sns.histplot(df["Annual Income (k$)"], kde=True, ax=axes[1], color="seagreen")
axes[1].set_title("Annual Income Distribution")
sns.histplot(df["Spending Score (1-100)"], kde=True, ax=axes[2], color="indianred")
axes[2].set_title("Spending Score Distribution")
plt.tight_layout()
plt.savefig("images/eda_distributions.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Gender")
plt.title("Income vs Spending Score")
plt.tight_layout()
plt.savefig("images/income_vs_spending.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 3. FEATURE SELECTION + SCALING
# ---------------------------------------------------------
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 4. FIND OPTIMAL K -- Elbow Method + Silhouette Score
# ---------------------------------------------------------
inertias = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_range), inertias, marker="o")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")

axes[1].plot(list(K_range), sil_scores, marker="o", color="darkorange")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Analysis")
plt.tight_layout()
plt.savefig("images/optimal_k.png", dpi=150)
plt.close()

best_k = list(K_range)[int(np.argmax(sil_scores))]
print(f"\nSuggested optimal K (by silhouette score): {best_k}")

# You can override this manually after inspecting the elbow chart, e.g. OPTIMAL_K = 5
OPTIMAL_K = 5

# ---------------------------------------------------------
# 5. FIT FINAL K-MEANS MODEL
# ---------------------------------------------------------
kmeans = KMeans(n_clusters=OPTIMAL_K, init="k-means++", n_init=10, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print(f"\nFinal silhouette score (K={OPTIMAL_K}):",
      round(silhouette_score(X_scaled, df["Cluster"]), 3))

# ---------------------------------------------------------
# 6. CLUSTER PROFILING
# ---------------------------------------------------------
profile = df.groupby("Cluster")[features].mean().round(1)
profile["Count"] = df["Cluster"].value_counts().sort_index()
print("\nCluster Profiles:\n", profile)
profile.to_csv("data/cluster_profiles.csv")

# ---------------------------------------------------------
# 7. VISUALIZE CLUSTERS
# ---------------------------------------------------------
plt.figure(figsize=(7, 6))
sns.scatterplot(
    data=df, x="Annual Income (k$)", y="Spending Score (1-100)",
    hue="Cluster", palette="tab10", s=70
)
plt.title(f"Customer Segments (K-Means, K={OPTIMAL_K})")
plt.tight_layout()
plt.savefig("images/clusters_income_spending.png", dpi=150)
plt.close()

fig = sns.pairplot(df, vars=features, hue="Cluster", palette="tab10")
fig.savefig("images/clusters_pairplot.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 8. SAVE MODEL + SCALER + LABELED DATA (for the Streamlit app)
# ---------------------------------------------------------
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
df.to_csv("data/segmented_customers.csv", index=False)

print("\nSaved: models/kmeans_model.pkl, models/scaler.pkl, data/segmented_customers.csv")
print("Saved charts to images/")
