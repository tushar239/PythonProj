'''
From chatgpt
Here’s a complete Python script you can run locally to visualize SVM decision boundaries
with different kernels (linear, RBF, and polynomial) and see how many support vectors
each model uses:

What You'll Learn by Running This:
- How different SVM kernels separate the same dataset
- The effectiveness of linear vs. non-linear kernels
- How many support vectors are used by each model (shown in the plot title)

Different kernels:
https://www.geeksforgeeks.org/machine-learning/major-kernel-functions-in-support-vector-machine-svm/
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Generate non-linear dataset (circles)
X, y = datasets.make_circles(n_samples=200, factor=0.5, noise=0.05, random_state=42)
X = StandardScaler().fit_transform(X)

# Train SVM models with different kernels
svm_linear = SVC(kernel='linear', C=1.0)
svm_rbf = SVC(kernel='rbf', gamma='auto', C=1.0)
svm_poly = SVC(kernel='poly', degree=3, gamma='auto', C=1.0)

svm_linear.fit(X, y)
svm_rbf.fit(X, y)
svm_poly.fit(X, y)

# Plotting function
def plot_svm(model, X, y, title, ax):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    ax.scatter(X[model.support_, 0], X[model.support_, 1], s=100,
               facecolors='none', edgecolors='black', label='Support Vectors')
    ax.set_title(f"{title}\nSupport Vectors: {len(model.support_)}")
    ax.legend()

# Plot all three SVMs
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
plot_svm(svm_linear, X, y, "Linear Kernel", axs[0])
plot_svm(svm_rbf, X, y, "RBF Kernel", axs[1])
plot_svm(svm_poly, X, y, "Polynomial Kernel (deg=3)", axs[2])

plt.tight_layout()
plt.show()
