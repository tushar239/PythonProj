import numpy as np

# 1D array
a = np.array([1, 2, 3])

# 2D array
b = np.array([[1, 2], [3, 4]])

# Zeros, ones, full
np.zeros((2, 3))       # 2x3 array of zeros
np.ones((3, 1))        # 3x1 array of ones
np.full((2, 2), 5)     # 2x2 array filled with 5

# Identity and eye matrix
np.eye(3)              # 3x3 identity matrix
