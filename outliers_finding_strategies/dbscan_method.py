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
    #'age': [15, 16, 14, 17, 18, 16, 100,101,102,103,104] # here, two clusters will be formed, no outlier
})

X = df[['age']]

def find_eps(min_samples):
    from sklearn.datasets import make_blobs
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    import numpy as np

    # Create sample data
    #X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)

    # Step 1: Choose min_samples and compute k-distance
    k = min_samples  # min_samples
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    print(distances)
    '''
    [[ 0.  1.]
     [ 0.  0.]
     [ 0.  1.]
     [ 0.  1.]
     [ 0.  1.]
     [ 0.  0.]
     [ 0. 82.]]
    '''
    # Step 2: Get the distances to the k-th nearest neighbor
    k_distances = np.sort(distances[:, k - 1])  # kth column (0-indexed)
    print(k_distances) # [ 0.  0.  1.  1.  1.  1. 82.]

    # Step 3: Plot the k-distance graph
    plt.figure(figsize=(8, 5))
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot
    plt.plot(k_distances) # plot x and y using default line style and color
    # same as
    #plt.plot([0,1,2,3,4,5,6], k_distances)
    #plt.plot([0,1,2,3,4,5,6], k_distances, marker='o', linestyle='--', color='green', linewidth=2, label='Squared Values')

    plt.ylabel(f"{k}th Nearest Neighbor Distance")
    plt.xlabel("Points sorted by distance")
    plt.title("K-distance Graph to Estimate eps for DBSCAN")
    plt.grid(True)
    plt.show()


# eps - Radius to search for neighboring points
# min_samples - Minimum number of points required to form a dense region (cluster)
# Usually, min_samples = 2 * number_of_features, If unsure, try between 4 to 10.
find_eps(2)
eps = 5
#dbscan = DBSCAN(eps=5, min_samples=2)
dbscan = DBSCAN(eps=eps, min_samples=2)
df['outlier'] = dbscan.fit_predict(X)

print(df)
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