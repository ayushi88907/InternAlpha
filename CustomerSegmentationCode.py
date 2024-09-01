import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the dataset
customer_dataset = pd.read_csv("C:\\Users\\Acer\\OneDrive\\Desktop\\Mall_Customers.csv")

# Display the first few rows of the dataset
print(customer_dataset.head())

# Data Exploration: Distribution of Age, Annual Income, and Spending Score
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(customer_dataset['Age'], kde=True)
plt.title('Age Distribution')

plt.subplot(1, 3, 2)
sns.histplot(customer_dataset['Annual Income (k$)'], kde=True)
plt.title('Annual Income Distribution')

plt.subplot(1, 3, 3)
sns.histplot(customer_dataset['Spending Score (1-100)'], kde=True)
plt.title('Spending Score Distribution')

plt.show()

# Gender Distribution
plt.figure(figsize=(6, 6))
sns.countplot(x='Gender', data=customer_dataset)
plt.title('Gender Distribution')
plt.show()

# Feature Selection for Clustering
X = customer_dataset[['Annual Income (k$)', 'Spending Score (1-100)']]

# Using the Elbow Method to Find the Optimal Number of Clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Graph
plt.figure(figsize=(10,5))
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying KMeans to the Dataset
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Adding Cluster Information to the Original Data
customer_dataset['Cluster'] = y_kmeans

# Visualizing the Clusters
plt.figure(figsize=(10,5))
plt.scatter(X.values[y_kmeans == 0, 0], X.values[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X.values[y_kmeans == 1, 0], X.values[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X.values[y_kmeans == 2, 0], X.values[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X.values[y_kmeans == 3, 0], X.values[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X.values[y_kmeans == 4, 0], X.values[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Optional: Analyzing Clusters by Gender
sns.countplot(x='Cluster', hue='Gender', data=customer_dataset)
plt.title('Gender Distribution across Clusters')
plt.show()
