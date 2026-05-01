import numpy as np

'''
recasts an array to new shape without changing its data
Here, arange(sart,stop,step) will return 1D array.
If you want to reshape it into nD array, then you can use reshape.
'''
array = np.arange(1, 10, 1).reshape((3,3))
print(array)

'''
[[1 2 3]
 [4 5 6]
 [7 8 9]]
'''

array.reshape(-1)

# array = np.arange(1, 11, 1).reshape((3,3)) # cannot reshape array of size 10 into shape (3,3)

# Create a 1D array with 6 elements
arr = np.array([1, 2, 3, 4, 5, 6])

# Reshape to 2 rows and 3 columns
new_arr = arr.reshape(2, 3)

print(new_arr)
# Output:
# [[1 2 3]
#  [4 5 6]]

'''
Key Features
- Automatic Dimension Calculation (-1): 
If you don't know one of the dimensions, you can use -1. 
NumPy will automatically calculate the correct value based on the total elements.
Example: arr.reshape(3, -1) on 12 elements will create a 3x4 matrix.

Flattening an Array: 
You can collapse any multi-dimensional array back into a 1D array by using reshape(-1)

View vs. Copy: 
Reshaping usually returns a "view" of the original array, meaning changes to the reshaped array will affect the original data.

'''

# convert 1-D array into 3-D array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(2, 3, 2) # The outermost dimension will have 2 arrays that contains 3 arrays, each with 2 elements:
print(newarr)
'''
[[[ 1  2]
  [ 3  4]
  [ 5  6]]

 [[ 7  8]
  [ 9 10]
  [11 12]]]
'''


# Flattening the arrays
'''
Flattening array means converting a multidimensional array into a 1D array.
We can use reshape(-1) to do this.
'''
arr = np.array([[1, 2, 3], [4, 5, 6]])
newarr = arr.reshape(-1)
print(newarr) # [1 2 3 4 5 6]

newarr = arr.reshape(-1, 1)
print(newarr)
'''
[[1]
 [2]
 [3]
 [4]
 [5]
 [6]]
'''

newarr = arr.reshape(1, -1)
print(newarr) # [[1 2 3 4 5 6]]