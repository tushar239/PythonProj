import numpy as np

'''
append(array, values, axis) - it will add the new elements at the end of given array and will produce a new array

array - input array
values = array of values to be inserted
axis - axis along which values should be inserted

axis = 0 means x-axis(horizontal axis), 1 means y-axis(vertical axis)
if axis=none, then it will flatten 2D array to 1D array alon with new elements that you added.
'''

# Adding a new row
a = np.array([[1,2,3], [4,5,6], [7,8,9]])
b = np.append(a, [[10,11,14]], 0) # axis=0 for adding as a row

print(b)
'''
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 14]]
'''
c = np.array([10,11,14]).reshape((3, 1))
d = np.append(a, c, 1) # axis=1 for adding as a col
print(d)

'''
[[ 1  2  3 10]
 [ 4  5  6 11]
 [ 7  8  9 14]]
'''