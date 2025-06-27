'''
Bar plot vs Histogram plot:
Bar plots are used for comparing categorical data,
while Histograms are used for visualizing the distribution of continuous data.

Histogram: Shows the distribution of numerical data.
'''

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Generate some data
'''
np.random.normal
https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
Draws random samples from a normal (Gaussian) distribution, often called the bell curve 
loc - mean value
scale - standard deviation
size - total numbers to generate
'''
data = np.random.normal(loc=0, scale=1, size=1000)
print(data)

# Plot histogram with KDE
sns.histplot(data, kde=True, bins=30)
plt.title("Histogram with KDE")
plt.show()

###########################################
# np.random.randn(1000) will produce a one-dimensional NumPy array containing 1000 values,
# where each value is a random sample from a normal distribution
# with a mean of 0 and a standard deviation of 1.
data = np.random.randn(1000)
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram")
plt.show()
