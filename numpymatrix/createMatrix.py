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

print(a.shape)  # (3, 4)
# array.shape returns a tuple. So, you can access a tuple to know number of rows and cols
print(a.shape[0])  # gives number of rows - 3
print(a.shape[1])  # gives number of cols - 4

print(a.size)  # gives number of elements - 12
