# code is taken from chatgpt
'''
Unlike classification models, K-Means clustering is unsupervised,
so it doesn’t predict labels in the traditional sense.

However, once the model is trained (i.e., cluster centers are learned),
you can assign new/unseen data points to the nearest cluster — this is what K-Means "prediction" means.
'''

from sklearn.cluster import KMeans
import numpy as np

# Training data (6 points)
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# Fit KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X) # learns cluster centers from the training data.

# New data point to predict
new_data = np.array([[0, 3], [11, 1]])

# Predict cluster for new data
# Calculates the distance from each new point to the centroids
# Assigns the point to the nearest cluster.
predicted_clusters = kmeans.predict(new_data)

print("Cluster assignments for new data:", predicted_clusters)
'''
Cluster assignments for new data: [1 0]
'''

# Assign Group Names to Predicted Clusters
# K-Means does not give class labels (like "spam" or "not spam") — you interpret clusters manually after training.
cluster_names = {0: "Group A", 1: "Group B"}
named_predictions = [cluster_names[c] for c in predicted_clusters]

for i, group in enumerate(named_predictions):
    print(f"Point {new_data[i]} → {group}")
'''
Point [0 3] → Group B
Point [11  1] → Group A
'''