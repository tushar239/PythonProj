# code is taken from chatgpt

'''
Each point is plotted once for each cluster, with color and alpha (transparency) indicating membership strength.
A point that belongs mostly to cluster 0 will appear more solid in orange, and vice versa for blue.
Cluster centers are marked with black Xs.
'''
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Step 1: Create sample 2D data
X = np.array([
    [1, 2],
    [2, 2],
    [2, 3],
    [8, 7],
    [8, 8],
    [25, 80]
]).T  # Shape: (features, samples)

# Step 2: Apply fuzzy c-means clustering
n_clusters = 2
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X, c=n_clusters, m=2.0, error=0.005, maxiter=1000, init=None)

# Step 3: Plot
colors = ['orange', 'blue']
fig, ax = plt.subplots()

for i in range(X.shape[1]):
    for j in range(n_clusters):
        ax.plot(X[0, i], X[1, i], 'o',
                color=colors[j],
                alpha=u[j, i],  # Transparency = degree of membership
                markersize=10)

# Plot cluster centers
ax.plot(cntr[:, 0], cntr[:, 1], 'kx', markersize=15, label='Centers')

ax.set_title("Fuzzy C-Means Clustering\n(Marker transparency = membership strength)")
ax.legend()
plt.grid(True)
plt.show()
