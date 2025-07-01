import numpy as np
'''
a = np.array([[1, 2, 3], [4, 5, 6]])

a[0, 1]       # element at row 0, column 1
a[:, 1]       # all rows, column 1
a[1, :]       # row 1, all columns
a[0:2, 1:]    # slicing subarray
'''
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
a = X[:, 0] # 0th column values from all rows
print(a) # [ 1  2  2  8  8 25]
b = X[:, 1]  # 1st column values from all rows
print(b) # [ 2  2  3  7  8 80]
c = X[:, 0:2]
print(c)
'''
[[ 1  2]
 [ 2  2]
 [ 2  3]
 [ 8  7]
 [ 8  8]
 [25 80]]
'''
d = X[2:4, 0] # 0th column data from 2nd and 3rd row
print(d) # [2, 8]