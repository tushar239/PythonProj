import numpy as np

a = np.matrix("1,2,3,4;4,5,6,7;7,8,9,10")
print(a)
'''
[[ 1  2  3  4]
 [ 4  5  6  7]
 [ 7  8  9 10]]
'''

# Second way to create a matrix
b = np.matrix([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
print(b)
'''
[[ 1  2  3  4]
 [ 4  5  6  7]
 [ 7  8  9 10]]
'''

# Third way to create a matrix - Instead of using matrix class, use array method
c = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
print(c)
'''
[[ 1  2  3  4]
 [ 4  5  6  7]
 [ 7  8  9 10]]
'''

# Fourth way to create a matrix from a range of numbers
d = np.arange(11, 20, 1).reshape((3, 3))
print(d)
'''
[[11 12 13]
 [14 15 16]
 [17 18 19]]
'''
# Fifth way to create a matrix
e = np.matrix(np.arange(21, 30, 1)).reshape(3, 3)
print(e)
'''
[[21 22 23]
 [24 25 26]
 [27 28 29]]
'''

print(a.shape)  # (3, 4)
# array.shape returns a tuple. So, you can access a tuple to know number of rows and cols
print(a.shape[0])  # gives number of rows - 3
print(a.shape[1])  # gives number of cols - 4

print(a.size)  # gives number of elements - 12
