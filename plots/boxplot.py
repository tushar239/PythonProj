# code is taken from chatgpt
'''
What is a Box Plot (Box-and-Whisker)?
A box plot (also called a box-and-whisker plot) shows:
    Median (middle line)
    Quartiles (box edges)
    Minimum and maximum (whiskers)
    Outliers (dots)

Quartiles are values that divide a sorted dataset into four equal parts, each containing 25% of the data.
The Four Quartiles:
Name	Meaning
Q1	    First quartile (25th percentile) → 25% of data is below it
Q2      (Median) Second quartile (50th percentile) → middle of the data
Q3	    Third quartile (75th percentile) → 75% of data is below it
IQR	    Interquartile Range = Q3 - Q1 → middle 50% spread

It's useful for visualizing the distribution and
detecting skewness or outliers in data.

https://www.geeksforgeeks.org/data-visualization/creating-multiple-boxplots-on-the-same-graph-from-a-dictionary/
'''

import matplotlib.pyplot as plt
import numpy as np

# Generate some random data
'''
np.random.normal
https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
Draws random samples from a normal (Gaussian) distribution, often called the bell curve 
loc - mean value
scale - standard deviation
size - total numbers to generate
'''
np.random.seed(0)
data1 = np.random.normal(50, 10, 200)  # Mean=50, SD=10
data2 = np.random.normal(60, 15, 200)  # Mean=60, SD=15
data3 = np.random.normal(55, 5, 200)   # Mean=55, SD=5

# Combine data into a list
data = [data1, data2, data3]

# Create box plot
plt.boxplot(data, labels=["Group A", "Group B", "Group C"])
plt.title("Box Plot Example")
plt.ylabel("Values")
plt.grid(True)

plt.show()
