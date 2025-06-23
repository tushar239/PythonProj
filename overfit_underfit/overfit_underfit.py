# code is taken from chatgpt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Generate synthetic non-linear data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # sine wave + noise

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit three models
models = {
    "Underfit (Degree 1)": make_pipeline(PolynomialFeatures(1), LinearRegression()),
    "Good Fit (Degree 4)": make_pipeline(PolynomialFeatures(4), LinearRegression()),
    "Overfit (Degree 15)": make_pipeline(PolynomialFeatures(15), LinearRegression())
}

# 4. Evaluate and visualize
plt.figure(figsize=(15, 5))

for i, (label, model) in enumerate(models.items(), 1):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    # Print scores
    print(f"{label}:")
    print(f"  Train MSE: {train_mse:.4f}")
    print(f"  Test MSE:  {test_mse:.4f}\n")

    # Plot
    X_curve = np.linspace(0, 5, 100).reshape(-1, 1)
    y_curve = model.predict(X_curve)

    plt.subplot(1, 3, i)
    plt.scatter(X_train, y_train, color='gray', alpha=0.5, label='Train Data')
    plt.scatter(X_test, y_test, color='black', alpha=0.6, label='Test Data')
    plt.plot(X_curve, y_curve, label=label.split()[0], linewidth=2)
    plt.title(label)
    plt.legend()
    plt.grid(True)

plt.suptitle('Underfitting vs Good Fit vs Overfitting', fontsize=16)
plt.tight_layout()
plt.show()
