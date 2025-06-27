'''
heatmap: Displays matrix-like data with color.
'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

'''
np.random.rand(5, 5) generates a new NumPy array with a shape of (5, 5) and populates it with random floating-point numbers. 
These numbers are uniformly distributed between 0.0 (inclusive) and 1.0 (exclusive).
'''
data = np.random.rand(5, 5)
print(data)
sns.heatmap(data, annot=True, cmap='coolwarm')
plt.title("Heatmap")
plt.show()
