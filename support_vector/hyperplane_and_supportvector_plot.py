# code is taken from chatgpt
# https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 1. Load a simple dataset (Iris)
iris = datasets.load_iris()
print(iris)
X = iris.data[:, :2]  # Using only 2 features for easy visualization
y = (iris.target != 0) * 1  # Binary classification: 1 if not setosa, else 0
print(type(X)) # <class 'numpy.ndarray'>
print(type(y)) # <class 'numpy.ndarray'>

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train SVM model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)


# 4. Plot decision boundary
def plot_svm_decision_boundary(model, X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # Plot margin and decision boundary
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])

    plt.title("SVM Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# 5. Run the plot
import numpy as np

plot_svm_decision_boundary(model, X_train, y_train)
