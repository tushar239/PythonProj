import numpy as np

a = np.array([1, 2, 3])
b = np.array([[1],
              [2],
              [3]])

print(a + b)  # broadcasting: adds each element of a to each row of b
