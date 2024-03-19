import numpy as np

a = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

'''
delete(array, obj, axis) - deletes a row or col from an array

array - input array
obj - indicates array to be removed or its position
axis - axis along which values should be inserted

axis = 0 means x-axis(horizontal axis), 1 means y-axis(vertical axis)
'''
b = np.delete(a, 1, 0)
print(b)
'''
[[1 2 3]
 [7 8 9]]
'''

c = np.delete(a, 1, 1)
print(c)
'''
[[1 3]
 [4 6]
 [7 9]]
'''