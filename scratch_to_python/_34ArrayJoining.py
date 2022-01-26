import numpy as np

"""
Joining methods
---------------
concatenate
stack
hstack  -  hstack is same as concatenation with axis=1
vstack  -  vstack is same as concatenation with axis=0
dstack
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

arr1 = np.array([[1, 2],
                 [3, 4]])
arr2 = np.array([[5, 6],
                 [7, 8]])
arr = np.concatenate((arr1, arr2), axis=1)
print(arr)
s = np.sum(arr, axis=1)  # axis=1 means cutting 2-D array horizontally and taking sum of those elements
print(s)
"""
[[1 2 5 6]
 [3 4 7 8]]
 
[14 22]
"""

arr1 = np.array([[0, 1, 2],
                 [3, 4, 5]])
arr2 = np.array([[7, 8, 9],
                 [10, 11, 12]])
arr = np.stack((arr1, arr2))  # look at the difference in the result from concatenation and stacking
print(arr)
"""
[
    [[ 0  1  2]
    [ 3  4  5]]
    ,
    [[ 7  8  9]
    [10 11 12]]
]
"""
arr1 = np.array([[0, 1, 2],
                 [3, 4, 5]])
arr2 = np.array([[7, 8, 9],
                 [10, 11, 12]])
arr = np.stack((arr1, arr2), axis=1)
print(arr)

"""
[
    [[ 0  1  2]
    [ 7  8  9]]
    ,
    [[ 3  4  5]
    [10 11 12]]]
"""

arr1 = np.array([[0, 1, 2],
                 [3, 4, 5]])
arr2 = np.array([[7, 8, 9],
                 [10, 11, 12]])
arr = np.vstack((arr1, arr2))  # vstack is same as concatenation with axis=0
print(arr)
"""
[[ 0  1  2]
 [ 3  4  5]
 [ 7  8  9]
 [10 11 12]]
"""

arr1 = np.array([[0, 1, 2],
                 [3, 4, 5]])
arr2 = np.array([[7, 8, 9],
                 [10, 11, 12]])
arr = np.hstack((arr1, arr2))  # hstack is same as concatenation with axis=1
print(arr)
"""
[[ 0  1  2  7  8  9]
 [ 3  4  5 10 11 12]]
"""

arr1 = np.array([[0, 1, 2],
                 [3, 4, 5]])
arr2 = np.array([[7, 8, 9],
                 [10, 11, 12]])
arr = np.dstack((arr1, arr2)) # based on the depth, arrays will be joined
print(arr)
"""
[
 [[ 0  7]
  [ 1  8]
  [ 2  9]]
 ,
 [[ 3 10]
  [ 4 11]
  [ 5 12]]
]
"""
