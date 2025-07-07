# Step 1: Import libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Step 2: Load a sample dataset (Iris)
X, y = load_iris(return_X_y=True)

# Step 3: Try different values of k (number of neighbors)
k_range = range(1, 31)
cv_scores = []  # to store average accuracy for each k

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)

    # Step 4: 5-fold cross-validation
    # cv=5 splits data into 5 parts and runs training/testing 5 times
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')

    # Step 5: Store the average score for this k
    cv_scores.append(scores.mean())

# Step 6: Find the best k
optimal_k = k_range[np.argmax(cv_scores)]
print(f"✅ Optimal number of neighbors (k): {optimal_k}")
print(f"🔍 Best accuracy: {max(cv_scores):.4f}")

# Step 7: Plot accuracy vs k
plt.figure(figsize=(8, 5))
plt.plot(k_range, cv_scores, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('KNN Cross-Validation Accuracy for Different k')
plt.grid(True)
plt.show()
##################### code with dataframe ###################################
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Load dataset into a Pandas DataFrame
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Step 2: Define X (features) and y (target)
X = df.drop('target', axis=1)
y = df['target']

# Step 3: Try different values of k (number of neighbors)
k_range = range(1, 31)
cv_scores = []  # to store average accuracy for each k

for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k)

    # Step 4: 5-fold cross-validation
    # cv=5 splits data into 5 parts and runs training/testing 5 times
    scores = cross_val_score(knn_model, X, y, cv=5, scoring='accuracy')

    # Step 5: Store the average score for this k
    cv_scores.append(scores.mean())

# Step 6: Find the best k
optimal_k = k_range[np.argmax(cv_scores)]
print(f"✅ Optimal number of neighbors (k): {optimal_k}")
print(f"🔍 Best accuracy: {max(cv_scores):.4f}")

# Step 7: Plot accuracy vs k
plt.figure(figsize=(8, 5))
plt.plot(k_range, cv_scores, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('KNN Cross-Validation Accuracy for Different k')
plt.grid(True)
plt.show()