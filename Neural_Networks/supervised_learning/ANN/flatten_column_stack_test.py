import numpy as np

a = np.array([[1],
 [2],
 [3]])

b = [4, 5, 6]

a_1D_array = a.flatten() # converts 2-D array into 1-D array. [1, 2, 3]
# joining same shape and size arrays column-wise
result = np.column_stack((a,b)) # pass arrays within a tuple or list
print(result)
'''
[[1 4]
 [2 5]
 [3 6]]
'''