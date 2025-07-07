'''
https://medium.com/@haidarlina4/k-means-clustering-with-scikit-learn-70c805230646
https://www.w3schools.com/python/python_ml_k-means.asp
'''
# code is taken from chatgpt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Sample 2D data
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# Apply K-means with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

# Get labels and centroids
labels = kmeans.labels_ # 0,1,... numbers are given to groups
centroids = kmeans.cluster_centers_

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red', label='Centroids')
plt.title("K-Means Clustering")
plt.legend()
plt.show()
