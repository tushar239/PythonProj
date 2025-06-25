# code is taken from chatgpt
from sklearn.cluster import KMeans
import numpy as np

##################### Using 2D array #######################
print("########## Using 2D array #########")
# Example 2D data
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# Apply KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

# Get numeric labels (0 or 1)
labels = kmeans.labels_

# Define group names for clusters
cluster_names = {
    0: "Group A",
    1: "Group B"
}

# Convert numeric labels to named groups
named_groups = [cluster_names[label] for label in labels]
print(named_groups)

for i, group in enumerate(named_groups):
    print(f"Data point {X[i]} belongs to {group}")

#################### Using Dataframe ############################
print("######### Using Dataframe ##########")
import pandas as pd

# Create DataFrame from data
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])

# Add numeric cluster labels
df["Cluster"] = kmeans.labels_ # adding a new column in dataframe for labels

# Map cluster labels to names
cluster_names = {0: "Group A", 1: "Group B"}
df["Group Name"] = df["Cluster"].map(cluster_names)

print(df)


