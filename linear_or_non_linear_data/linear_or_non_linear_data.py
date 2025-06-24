# Find out whether the data is linear or non-linear
# code is taken from chatgpt

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


# Study hours (independent variable)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
# Corresponding scores (dependent variable)
y = np.array([35, 45, 50, 55, 60, 65, 70, 75, 85, 95])

plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Scatter Plot")
plt.grid(True)
plt.show()
'''
Straight-line trend → Linear
Curved, wave-like, or U-shaped trend → Non-linear
'''

# Reshape if X is 1D
#X_reshaped = X.reshape(-1, 1)
model = LinearRegression()
#model.fit(X_reshaped, y)
model.fit(X, y)
#y_pred = model.predict(X_reshaped)
y_pred = model.predict(X)

# Plot residuals
residuals = y - y_pred
plt.scatter(X, residuals)
plt.axhline(0, color='red')
plt.title("Residual Plot")
plt.show()
'''
Random scatter around 0 → Linear
Pattern (curve, funnel, wave) → Non-linear
'''

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline

# Linear
#linear_r2 = r2_score(y, model.predict(X_reshaped))
linear_r2 = r2_score(y, model.predict(X))

# Polynomial
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
#poly_model.fit(X_reshaped, y)
poly_model.fit(X, y)
#poly_r2 = r2_score(y, poly_model.predict(X_reshaped))
poly_r2 = r2_score(y, poly_model.predict(X))

print("Linear R²:", linear_r2) # 0.9828258010076192
print("Polynomial R²:", poly_r2) # 0.9867967140694414
'''
If polynomial R² is much higher than linear R² → Data is non-linear
If both are close → Data is linear
'''

from scipy.stats import pearsonr

r, _ = pearsonr(X.flatten(), y) # returns tuple
print("Pearson correlation coefficient:", r) # 0.9913757113262456
'''
r ≈ ±1: Strong linear correlation
r ≈ 0: Likely non-linear or no correlation
'''