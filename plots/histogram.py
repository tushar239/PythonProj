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
