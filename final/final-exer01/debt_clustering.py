import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv("data/debtData.csv")

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
import matplotlib.pyplot as plt

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

# Visualize the clusters (using pairplots or any other visualization technique)
import seaborn as sns

sns.pairplot(data=data, hue='Cluster')
plt.show()

# Analyze cluster characteristics
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_df = pd.DataFrame(cluster_centers, columns=features)
print(cluster_df)
