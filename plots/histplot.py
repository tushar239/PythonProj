# histplot/distplot :
# Used to visualize the distribution of a numeric variable.
# Distplot is a combination of a histogram with a line (density plot) on it.

# histogram vs hist/dist plot:
# histogram plot can not have kde. hist/dist plot can have kde.

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# np.random.randn(1000) will produce a one-dimensional NumPy array containing 1000 values,
# where each value is a random sample from a normal distribution
# with a mean of 0 and a standard deviation of 1.
data = np.random.randn(1000)

'''
KDE provides a smoother visualization of data distribution compared to histograms, 
which can be particularly useful for identifying subtle patterns like skewness, 
multimodality, or outliers.

In Seaborn, KDE stands for Kernel Density Estimate. It refers to a non-parametric method used to estimate the probability density function (PDF) of a continuous random variable.
Instead of displaying data in discrete bins like a histogram, a KDE plot generates a smooth, continuous curve that represents the distribution of observations in a dataset. 
This curve is derived by placing a "kernel" (typically a Gaussian function) over each data point and then summing these kernels to create a continuous estimate of the underlying probability density.
'''
sns.histplot(data, kde=True, bins=30)  # newer version; distplot is deprecated
plt.title("Distribution Plot with KDE")
plt.show()
