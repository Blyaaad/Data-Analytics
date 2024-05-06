import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, calinski_harabasz_score, davies_bouldin_score

# Load the dataset
data = pd.read_csv("data/debtData.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Display the first few rows of the dataset
print(data.head())

## Preprocessing
# Remove irrelevant columns (CustomerID)
data = data.drop(columns=['CustomerID'])

# Handling missing values (if any)
data.fillna(data.mean(), inplace=True)

# Scale the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Choose features for clustering
features = ['Age',
            'EducationLevel',
            'YearsEmployed',
            'Income',
            'CardDebt',
            'OtherDebt',
            'Defaulted',
            'DebtIncomeRatio'
            ]
X = scaled_data

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Choose the optimal number of clusters (e.g., 3) using the KMeans
n_clusters = 3

# Train the KMeans clustering model
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
kmeans.fit(X)

# Add cluster labels to the original dataframe
data['Cluster'] = kmeans.labels_

# Evaluate the clustering model using silhouette score
silhouette_avg = silhouette_score(X, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)

# Additional performance metrics
rand_index = adjusted_rand_score(data['Cluster'], kmeans.labels_)
mutual_info = adjusted_mutual_info_score(data['Cluster'], kmeans.labels_)
calinski_harabasz = calinski_harabasz_score(X, kmeans.labels_)
davies_bouldin = davies_bouldin_score(X, kmeans.labels_)

print("Adjusted Rand Index:", rand_index)
print("Adjusted Mutual Information:", mutual_info)
print("Calinski-Harabasz Index:", calinski_harabasz)
print("Davies-Bouldin Index:", davies_bouldin)

# Visualize the silhouette plot
silhouette_values = silhouette_samples(X, kmeans.labels_)
y_lower = 10

fig, ax = plt.subplots()
ax.set_title("Debt Silhouette Plot")
ax.set_xlabel("Silhouette coefficient values")
ax.set_ylabel("Cluster Label")

for i in range(n_clusters):
    cluster_silhouette_values = silhouette_values[kmeans.labels_ == i]
    cluster_silhouette_values.sort()
    cluster_size = cluster_silhouette_values.shape[0]
    y_upper = y_lower + cluster_size
    color = plt.cm.nipy_spectral(float(i) / n_clusters)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * cluster_size, str(i))
    y_lower = y_upper + 10


ax.axvline(x=silhouette_avg, color="red", linestyle="--")
ax.annotate('Average Silhouette\nscore = {:.2f}'.format(silhouette_avg),
            xy=(silhouette_avg, y_lower),
            xytext=(silhouette_avg + 0.1, y_lower),
            ha='center',
            va='center',
            bbox=dict(boxstyle="round", alpha=0.1),
            arrowprops=dict(arrowstyle="->"))

ax.set_yticks([])

plt.show()

# Visualize the clusters (using pairplots or any other visualization technique)
sns.pairplot(data=data, hue='Cluster')
plt.show()

# Analyze cluster characteristics
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_df = pd.DataFrame(cluster_centers, columns=features)

print(cluster_df.to_string(index=False))
