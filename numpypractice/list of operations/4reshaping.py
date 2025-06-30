import numpy as np

a = np.array([[1, 2], [3, 4]])
a.reshape((4, 1))      # reshape to 4x1
a.flatten()            # flatten to 1D
a.T                    # transpose of matrix. same as a.transpose()

b = a.reshape((-1,1)) # converts into 1 col 2-D array. -1 means unknown dimension.
print(b)
'''
[[1]
 [2]
 [3]
 [4]]
'''

b = a.reshape((-1,1,1))
print(b)
'''
[[[1]]

 [[2]]

 [[3]]

 [[4]]]
'''

# flattens into 1-D array
b = a.flatten()
print(b) # [1 2 3 4]

# same as transpose() - rows become cols and vice-a-versa
b = a.T
print(b)
'''
[[1 3]
 [2 4]]
'''