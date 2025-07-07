# Use LOF (Local Outlier Factor) method to find outliers when
# you want to find outliers in clusters or variable densities
# When you need to consider local context rather than global statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# Create synthetic data with outliers
X = np.array([[1, 2], [2, 1.5], [2, 2], [8, 8], [8.5, 8], [25, 80]])  # Last point is likely an outlier

# Fit LOF model. n_neighbors is same has number of neighbours in KNN algorithm.
lof = LocalOutlierFactor(n_neighbors=2)  # n_neighbors typically 10-20 for real data
y_pred = lof.fit_predict(X)  # -1 = outlier, 1 = inlier
scores = -lof.negative_outlier_factor_  # Higher = more likely to be outlier

print(type(scores)) # <class 'numpy.ndarray'>
print(scores) # [0.9736068  0.9736068  1.05572809 4.37537064 4.37537064 8.53064379]

# Display outlier scores
for i, score in enumerate(scores):
    print(f"Point {X[i]} -> LOF Score: {score:.2f}, Outlier: {y_pred[i] == -1}")

# Plot
# c= color. Here, color is y_pred, so -1 or 1
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', s=100, edgecolor='k')
plt.title("Local Outlier Factor (LOF) Result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
