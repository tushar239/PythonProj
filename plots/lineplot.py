# Shows trends over time or continuous data.

import matplotlib.pyplot as plt
import numpy as np

# numpy.linspace() is a function in the NumPy library of Python that generates an array of evenly spaced numbers over a specified interval.
# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
#x = np.linspace(0, 10, 100)
#y = np.sin(x)

x = [1,2,3,4,5]
y = [1,4,9,16,25]

# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot
plt.plot(x, y) # plot x and y using default line style and color
# plot(x, y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
plt.title("Line Plot")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show()
