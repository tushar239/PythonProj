from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # noisy sine wave

# Train SVR
model = SVR(kernel='rbf', C=100, epsilon=0.1)
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Plot
plt.scatter(X, y, color='gray', label='Actual Data')
plt.plot(X, y_pred, color='red', label='SVR Prediction')
plt.title('SVR Regression Example')
plt.legend()
plt.show()
