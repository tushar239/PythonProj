import numpy as np

"""
Joining methods
---------------
concatenate
stack
hstack
vstack
"""


# Concatenation
# https://www.youtube.com/watch?v=GfP5Zuioya0

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print(arr)


# https://www.youtube.com/watch?v=KKCBdIP8H88

arr1 = np.array([[1, 2],
                 [3, 4]])
arr2 = np.array([[5, 6],
                 [7, 8]])
arr = np.concatenate((arr1, arr2), axis=0)
print(arr)
s = np.sum(arr, axis=0)  # axis=0 means cutting 2-D array vertically and taking sum of those elements  = [1+3+5+7   2+4+6+8] = [16 20]
print(s)
"""
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
 
[16 20]
"""


arr = np.concatenate((arr1, arr2), axis=1)
print(arr)
s = np.sum(arr, axis=1)  # axis=1 means cutting 2-D array horizontally and taking sum of those elements
print(s)
"""
[[1 2 5 6]
 [3 4 7 8]]
 
[14 22]
"""


