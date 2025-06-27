# Pie chart: Shows parts of a whole.

import matplotlib.pyplot as plt
import numpy as np

labels = ['Python', 'Java', 'C++']
sizes = [45, 30, 25]

plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Pie Chart")
plt.show()
