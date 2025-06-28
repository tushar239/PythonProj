# Scatter plot: Shows the relationship between two numeric continuous variables.

import matplotlib.pyplot as plt
import numpy as np

'''
np.random.rand(100)
it will create 100 random numbers between 0 and 1.
These numbers are sampled from a uniform distribution over the half-open interval [0.0, 1.0), 
meaning they are greater than or equal to 0.0 and strictly less than 1.0.

np.random.normal
https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
Draws random samples from a normal (Gaussian) distribution, often called the bell curve 
loc - mean value
scale - standard deviation
size - total numbers to generate
'''

x = np.random.rand(100)
y = x + np.random.normal(0, 0.1, 100)
print(x)
print(y)

plt.scatter(x, y)
plt.title("Scatter Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
##################################################33
# Sample data
x = [1, 2, 3, 4, 5]  # X-axis values
y = [2, 4, 5, 4, 5]  # Y-axis values

# Create scatter plot
#plt.scatter(x, y)

#Change color and marker
#plt.scatter(x, y, color='red', marker='x')

# Add size to points (observations)
sizes = [20, 40, 60, 80, 100]
# plt.scatter(x, y, s=sizes, color='red', marker='x')

colors = [1, 2, 3, 4, 5]  # Used to color-code points
plt.scatter(x, y, c=colors, cmap='viridis', marker='x', s=sizes)

#  to show regression lines or model fits.
plt.plot(x, y) # line connecting observations

#for better readability
plt.grid(True)

# Add labels and title
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Scatter Plot Example")

# Use legend() when plotting multiple scatter plots.
# plt.legend()

# Show plot
plt.show()

#################################################

import seaborn as sns

df = sns.load_dataset("tips")
print(df.info())
plt.scatter(x="day", y="total_bill", data=df)
plt.xlabel("day")
plt.ylabel("total bill")
plt.title("Scatter Plot")
plt.show()