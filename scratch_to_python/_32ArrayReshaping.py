import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(4, 3)

print(newarr)
print(len(newarr))  # 4
print(len(newarr[0]))  # 3
"""
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
"""

newarr = arr.reshape(2, 6)
print(newarr)
print(len(newarr))  # 2
print(len(newarr[0]))  # 6
"""
[[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]]
"""

newarr = arr.reshape(2, 3, 2)

print(newarr)
print(len(newarr))  # 2
print(len(newarr[0]))  # 3
"""
[
    [
        [ 1  2]
        [ 3  4]
        [ 5  6]
    ]

    [
        [ 7  8]
        [ 9 10]
        [11 12]
    ]
]
"""

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(arr.shape) # (2, 4)


arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr)  # [[[[[1 2 3 4]]]]]
print('shape of array :', arr.shape)  # (1, 1, 1, 1, 4)
