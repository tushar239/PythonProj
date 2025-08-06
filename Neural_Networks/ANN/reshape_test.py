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