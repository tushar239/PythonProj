'''
Bar plot vs Histogram plot:
Bar plots are used for comparing categorical data,
while histograms are used for visualizing the distribution of continuous data.

Bar plot: Compares categories or values.
'''

import matplotlib.pyplot as plt
import numpy as np

categories = ['A', 'B', 'C']
values = [10, 24, 18]

plt.bar(categories, values)
plt.title("Bar Plot")
plt.show()
