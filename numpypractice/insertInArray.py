import numpy as np
'''
insert(array, obj, values, axis) adds values at a given position and axis in an array.

array - input array
obj - index position
values = array of values to be inserted
axis - axis along which values should be inserted

axis = 0 means x-axis(horizontal axis), 1 means y-axis(vertical axis)
if axis=none, then it will flatten 2D array to 1D array alon with new elements that you added.
'''

a = np.array([[1,2,3], [4,5,6], [7,8,9]])
b = np.insert(a, 1, [10,11,12], 0)

print(b)
'''
[[ 1  2  3]
 [10 11 12]
 [ 4  5  6]
 [ 7  8  9]]
'''

c = np.array([10,11,12])
print(c)
d = np.insert(a, 1, c, 1)
print(d)
'''
[[ 1 10  2  3]
 [ 4 11  5  6]
 [ 7 12  8  9]]
'''