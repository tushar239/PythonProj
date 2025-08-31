import numpy as np

a = np.array([[1,2,3], [4,5,6], [7,8,9]])

# Both syntaxes work to access the elements from 2D array
print(a[0][1])
print(a[0,1])


print(a[2][2])
print(a[2,2])

'''
slicing (extracting) - It is just like slicing a python list
syntax for 1D array - array[start : stop : step]
syntax for 2D array - array[row start : row stop : row step, col start : col stop : col step]

default start = 0
default stop = last index + 1
default step = 1
'''

a = np.array([[1,2,3], [4,5,6], [7,8,9]])
# retrieving sub-array
subArray = a[1:3] # default step=1
print(subArray)
'''
[[4 5 6]
 [7 8 9]]
'''


subArray = a[::2]  # default start=0, stop=3
print(subArray)
'''
[[1 2 3]
 [7 8 9]]
'''

# slicing (extracting column from extracted rows)
subArray = a[1:3, 0]
print(subArray) # [4 7]
'''
a[1:3] extracts
[ 
    [4,5,6],
    [7,8,9]
]
a[1:3, 0] extracts 0th column
    [4,7]
'''

subArray = a[1:3, 0:2]
print(subArray)
'''
[[4 5]
 [7 8]]
'''

#column_stack - converts 1-D arrays as columns of 2-D array
# numpy's column_stack - https://www.geeksforgeeks.org/numpy-column_stack-in-python/
# input array
in_arr1 = np.array((1, 2, 3))
print("1st Input array : \n", in_arr1)

in_arr2 = np.array((4, 5, 6))
print("2nd Input array : \n", in_arr2)

# Stacking the two arrays
out_arr = np.column_stack((in_arr1, in_arr2))
print("Output stacked array:\n ", out_arr)
'''
[[1 4]
 [2 5]
 [3 6]]
'''

# concatinating two arrays (rowwise, columnwise)
import numpy as np

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