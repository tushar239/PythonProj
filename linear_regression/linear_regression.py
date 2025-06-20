# this example is taken from chatgpt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Study hours (independent variable)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
# Corresponding scores (dependent variable)
y = np.array([35, 45, 50, 55, 60, 65, 70, 75, 85, 95])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions on test data:", y_pred) # Predictions on test data: [84.91336634 41.68316832 66.38613861]

print("Mean Squared Error:", mean_squared_error(y_test, y_pred)) # Mean Squared Error: 4.310086020978356
print("R² Score:", r2_score(y_test, y_pred)) # R² Score: 0.9838371774213311

# Plot training data
plt.scatter(X_train, y_train, color='blue', label='Training data')
# Plot test data
plt.scatter(X_test, y_test, color='red', label='Test data')
# Plot regression line
plt.plot(X, model.predict(X), color='black', label='Regression line')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.title('Linear Regression Example')
plt.legend()
plt.grid(True)
plt.show()
