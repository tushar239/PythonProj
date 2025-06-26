from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Example 2D data
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])


inertia = []
for k in range(1, 6):
    model = KMeans(n_clusters=k)
    model.fit(X)
    inertia.append(model.inertia_)

plt.plot(range(1, 6), inertia, marker='o')
plt.xlabel("Number of Clusters K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
