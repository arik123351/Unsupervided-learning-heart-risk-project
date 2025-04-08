import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load Data
data = pd.read_csv('/Users/arielshamis/Downloads/framingham_heart_study.csv')  # Replace with actual filename

# Preprocessing: Drop non-numeric & target columns for clustering
features = data.drop(columns=["TenYearCHD"])  # Remove target variable
data_cleaned = features.dropna()  # Drop missing values

# Standardize Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_cleaned)

# Determine Optimal Clusters using Elbow Method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# Apply K-Means with chosen K (Assume k=3 based on Elbow Graph)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
data_cleaned['Cluster'] = kmeans.fit_predict(data_scaled)

# Reduce Dimensions for Visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)
data_cleaned['PCA1'] = pca_result[:, 0]
data_cleaned['PCA2'] = pca_result[:, 1]

# Scatter Plot of Clusters
plt.figure(figsize=(8, 6))
for cluster in range(2):
    subset = data_cleaned[data_cleaned['Cluster'] == cluster]
    plt.scatter(subset['PCA1'], subset['PCA2'], label=f'Cluster {cluster}')
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("K-Means Clustering Visualization")
plt.legend()
plt.show()

# Analyze Cluster Characteristics
cluster_means = data_cleaned.groupby('Cluster').mean()
print("Cluster Analysis:\n", cluster_means)
