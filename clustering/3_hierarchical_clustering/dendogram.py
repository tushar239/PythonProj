import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Sample data
X = np.array([[1, 2], [2, 3], [3, 4], [10, 10], [11, 11]])

# Perform hierarchical clustering
linked = linkage(X, method='ward')  # Try 'single', 'average', etc.

# Plot dendrogram
plt.figure(figsize=(8, 4))
dendrogram(linked, labels=[f'P{i}' for i in range(len(X))])
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
