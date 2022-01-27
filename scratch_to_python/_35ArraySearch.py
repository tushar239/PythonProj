import numpy as np

arr = np.array([1, 2, 3, 4, 5, 4, 4])

indices = np.where(arr == 4)
print(indices)  # (array([3, 5, 6], dtype=int64),)
print(type(indices))  # <class 'tuple'>
print(type(indices[0]))  # <class 'numpy.ndarray'>
print(indices[0])  # [3 5 6]

# find indices of odd numbers
indices = np.where(arr%2 == 1)
print(indices[0])  # [0 2 4]

# Searching in a sorted array where a given element can be inserted in given array.
# It uses Binary Search algorithm to do that
arr = np.array([6, 7, 8, 9])
index = np.searchsorted(arr, 10)
print(index)



