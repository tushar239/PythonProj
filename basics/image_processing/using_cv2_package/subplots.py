# https://www.w3schools.com/python/matplotlib_subplot.asp

import matplotlib.pyplot as plt
import numpy as np

#plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

'''
The subplot() function takes three arguments that describes the layout of the figure.

The layout is organized in rows and columns, which are represented by the first and second argument.

The third argument represents the index of the current plot.

'''
# the figure has 2 rows, 1 column, and this plot is the first plot.
plt.subplot(2, 1, 1)
plt.plot(x,y)

#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

# the figure has 2 rows, 1 column, and this plot is the second plot.
plt.subplot(2, 1, 2)
plt.plot(x,y)

plt.show()