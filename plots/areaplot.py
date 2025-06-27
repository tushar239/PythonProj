# Area plot: Cumulative or stacked version of line plots.

import matplotlib.pyplot as plt
import numpy as np


days = [1, 2, 3, 4, 5]
sales = [100, 120, 90, 130, 160]

plt.fill_between(days, sales, color='lightblue')
plt.title("Area Plot")
plt.xlabel("Day")
plt.ylabel("Sales")
plt.show()
