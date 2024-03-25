import numpy as np

a = np.matrix("1,2,3,4;4,5,6,7;7,8,9,10")

'''
inserts a row or col in a matrix

numpy.insert(matrix, obj, values, axis)

matrix - input matrix
obj - index position
values - array or matrix of values to be inserted
axis - axis along which values should be inserted

axis = 0 means x-axis(horizontal axis), 1 means y-axis(vertical axis)
'''

b = np.insert(arr=a, axis=0, obj=1, values=np.matrix("10,11,12,13;13,14,15,16"))
print(b)

'''
[[ 1  2  3  4]
 [10 11 12 13]
 [13 14 15 16]
 [ 4  5  6  7]
 [ 7  8  9 10]]
'''

b = np.insert(arr=a, axis=0, obj=1, values=[10, 11, 12, 13])
print(b)

'''
[[ 1  2  3  4]
 [10 11 12 13]
 [ 4  5  6  7]
 [ 7  8  9 10]]
'''
