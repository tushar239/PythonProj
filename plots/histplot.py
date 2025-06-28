# histplot/distplot :
# Used to visualize the distribution of a numeric variable.
# Distplot is a combination of a histogram with a line (density plot) on it.

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# np.random.randn(1000) will produce a one-dimensional NumPy array containing 1000 values,
# where each value is a random sample from a normal distribution
# with a mean of 0 and a standard deviation of 1.
data = np.random.randn(1000)

sns.histplot(data, kde=True, bins=30)  # newer version; distplot is deprecated
plt.title("Distribution Plot with KDE")
plt.show()
