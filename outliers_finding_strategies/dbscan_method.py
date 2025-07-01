'''
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN groups together points that are close to each other (i.e., dense regions) and marks points in sparse regions as outliers.

eps	- Radius to search for neighboring points
min_samples	- Minimum number of points required to form a dense region (cluster)
Core Point - A point with at least min_samples neighbors within eps
Border Point - Fewer neighbors than min_samples, but in the neighborhood of a core
Noise Point	- Not a core or border — outlier

Tips for Use
Choosing eps is critical! Use k-distance graph to tune it.
Too small eps: Too many outliers.
Too large eps: All points may be one big cluster.
'''

from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample DataFrame
df = pd.DataFrame({
    'age': [15, 16, 14, 17, 18, 16, 100]  # 100 is an outlier
})

X = df[['age']]
dbscan = DBSCAN(eps=5, min_samples=2)
df['outlier'] = dbscan.fit_predict(X)

# Outliers are labeled as -1
print(df[df['outlier'] == -1])
'''
   age  outlier
6  100       -1
'''
##########################################################33
# Sample dataset with one outlier
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])  # last point is likely outlier

db = DBSCAN(eps=3, min_samples=2).fit(X)
labels = db.labels_  # -1 indicates outliers

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', s=100)
plt.title("DBSCAN Clustering with Outlier Detection")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# View which points are outliers
print("Outlier indices:", np.where(labels == -1)[0])