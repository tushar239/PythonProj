import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(4, 3)  # convert original array in new array with 4 rows and 3 columns

print(newarr)
print(len(newarr))  # 4
print(len(newarr[0]))  # 3
"""
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
"""

newarr = arr.reshape(2, 6)  # convert original array in new array with 2 rows and 6 columns
print(newarr)
print(len(newarr))  # 2
print(len(newarr[0]))  # 6
"""
[[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]]
"""

newarr = arr.reshape(2, 3, 2)  # convert original array in new array with 2 rows, 3 columns, group 2 elements in each row

print(newarr)
print(len(newarr))  # 2
print(len(newarr[0]))  # 3
"""
[
    [
        [ 1  2] [ 3  4] [ 5  6]
    ]

    [
        [ 7  8] [ 9 10] [11 12]
    ]
]
"""

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8]])

print(arr.shape)  # (2, 4)  - tuple of number of outer array elements(2), number of each inner array elements(4)


arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr)  # [[[[[1 2 3 4]]]]]
print('shape of array :', arr.shape)  # (1, 1, 1, 1, 4)
