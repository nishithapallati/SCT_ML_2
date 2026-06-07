# =====================================================
# Customer Segmentation using K-Means Clustering
# Dataset: Mall_Customers.csv
# =====================================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("dataset/Mall_Customers.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================================
# Data Visualization
# =====================================================

# Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Annual Income Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Annual Income (k$)"], bins=20, kde=True)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Count")
plt.show()

# Spending Score Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Spending Score (1-100)"], bins=20, kde=True)
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Count")
plt.show()

# =====================================================
# Feature Selection
# =====================================================

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# =====================================================
# Elbow Method
# =====================================================

wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# =====================================================
# Apply K-Means
# =====================================================

kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

# Add Cluster Column
df["Cluster"] = clusters

# =====================================================
# Visualize Clusters
# =====================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    X.iloc[clusters == 0, 0],
    X.iloc[clusters == 0, 1],
    s=60,
    label="Cluster 1"
)

plt.scatter(
    X.iloc[clusters == 1, 0],
    X.iloc[clusters == 1, 1],
    s=60,
    label="Cluster 2"
)

plt.scatter(
    X.iloc[clusters == 2, 0],
    X.iloc[clusters == 2, 1],
    s=60,
    label="Cluster 3"
)

plt.scatter(
    X.iloc[clusters == 3, 0],
    X.iloc[clusters == 3, 1],
    s=60,
    label="Cluster 4"
)

plt.scatter(
    X.iloc[clusters == 4, 0],
    X.iloc[clusters == 4, 1],
    s=60,
    label="Cluster 5"
)

# Centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=250,
    marker="X",
    color="black",
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# Cluster Analysis
# =====================================================

print("\nCustomer Count in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())

print("\nCluster Summary:")
print(df.groupby("Cluster").mean(numeric_only=True))

# =====================================================
# Save Output
# =====================================================

df.to_csv("customer_segments.csv", index=False)

print("\ncustomer_segments.csv saved successfully!")

# =====================================================
# Customer Segment Interpretation
# =====================================================

print("\nPossible Cluster Interpretation:")
print("""
Cluster 0 -> Average Income, Average Spending
Cluster 1 -> High Income, High Spending
Cluster 2 -> Low Income, High Spending
Cluster 3 -> High Income, Low Spending
Cluster 4 -> Low Income, Low Spending

(Note: Actual cluster numbers may vary each run.)
""")