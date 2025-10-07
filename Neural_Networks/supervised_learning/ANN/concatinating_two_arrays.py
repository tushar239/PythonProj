# concatinating two arrays (rowwise, columnwise)
import numpy as np

# Two 2D arrays
a = np.array([[1, 2],
              [3, 4]])

b = np.array([[5, 6],
              [7, 8]])

# Row-wise concatenation
result = np.concatenate((a, b), axis=0)
print(result)

# Alternative
result = np.vstack((a, b))
print(result)

# Alternative
result = np.r_[a, b]
print(result)
'''
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
'''

# Column-wise concatenation
# Requirement: The number of rows must match (same shape along axis=0).
result = np.concatenate((a, b), axis=1)
print(result)

# Alternative
result = np.hstack((a, b))
print(result)

# Alternative
result = np.c_[a, b]
print(result)

'''
[[1 2 5 6]
 [3 4 7 8]]
'''