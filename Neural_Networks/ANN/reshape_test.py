import numpy as np

a = np.array([[False],
 [False],
 [False]])
#print(a)
a_2D_array = a.reshape(len(a),1) #reshape converts 1-D array to 2-D array
print(a_2D_array)
'''
[[False]
 [False]
 [False]]
'''

a_1D_array = a.reshape(-1) # same as a.flatten()
print(a_1D_array) # [False False False]

b= np.array([False, False, False, False, False, False])
b_1D_array = b.reshape(len(b),1) # reshape converts 1-D array to 2-D array
print(b_1D_array)
'''
[[False]
 [False]
 [False]
 [False]
 [False]
 [False]]
'''

'''
In Python's NumPy library, reshape(-1, 1) is used to transform an array into a column vector.
Here is a breakdown of what reshape(-1, 1) means:
-1: This placeholder tells NumPy to automatically calculate the size of that dimension based on the total number of elements in the array and the other specified dimensions. In this case, it will determine the number of rows needed to accommodate all elements.
1: This explicitly sets the second dimension (columns) to a size of 1.

When you apply array.reshape(-1, 1) to a 1D array or a multi-dimensional array, the result will be a 2D array with a single column, and the number of rows will be determined by the total number of elements in the original array.
'''
arr_1d = np.array([1, 2, 3, 4, 5, 6])
print("Original 1D array:")
print(arr_1d)
print("Shape:", arr_1d.shape)

# Reshape to a column vector
column_vector = arr_1d.reshape(-1, 1)
print("\nReshaped column vector:")
print(column_vector)
print("Shape:", column_vector.shape)
'''
Reshaped column vector:
[[1]
 [2]
 [3]
 [4]
 [5]
 [6]]
Shape: (6, 1)
'''

# A 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("\nOriginal 2D array:")
print(arr_2d)
print("Shape:", arr_2d.shape)

# Reshape to a column vector
column_vector_from_2d = arr_2d.reshape(-1, 1)
print("\nReshaped column vector from 2D array:")
print(column_vector_from_2d)
print("Shape:", column_vector_from_2d.shape)
'''
Reshaped column vector from 2D array:
[[1]
 [2]
 [3]
 [4]
 [5]
 [6]]
Shape: (6, 1)
'''