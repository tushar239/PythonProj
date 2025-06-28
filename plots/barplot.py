'''
Bar plot vs Histogram plot:
Bar plots are used for comparing categorical data,
while histograms are used for visualizing the distribution of continuous data.

Bar plot: Compares categories or values.
A Bar plot/Count plot that shows the number of occurrences of each category
'''

import matplotlib.pyplot as plt
import numpy as np

categories = ['A', 'B', 'C']
values = [10, 24, 18]

plt.bar(categories, values)
plt.title("Bar Plot")
plt.show()
##############################################
import seaborn as sns

# Load example dataset
df = sns.load_dataset("titanic")

# Bar plot of 'sex' column
# This will show how many passengers are male vs. female in the Titanic dataset.
category_counts = df['sex'].value_counts()
print(type(category_counts)) # <class 'pandas.core.series.Series'>

print(category_counts.values)
print(type(category_counts.values))

print(category_counts.index.values) # <class 'numpy.ndarray'>
print(type(category_counts.index.values)) # <class 'numpy.ndarray'>

plt.bar(category_counts.index.values, category_counts.values)
plt.title("Count of Passengers by Sex")
plt.show()