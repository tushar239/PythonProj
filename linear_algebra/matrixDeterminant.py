import numpy as np
# linalg is a package inside numpy package
import numpy.linalg as la
# How to calculate the determinant of a matrix?
# https://en.wikipedia.org/wiki/Determinant

a = np.matrix([[1, 2],
               [4, 5]])
result = la.det(a) # 1x5 - 2x4
print(result) # -2.9999999999999996

# creating 4 X 4 matrix
a = np.matrix("4,5,16,7;2,-3,2,3;3,4,5,6;4,7,8,9")
print(a)
'''
[[ 4  5 16  7]
 [ 2 -3  2  3]
 [ 3  4  5  6]
 [ 4  7  8  9]]
'''
result = la.det(a)
print(result) # 128.00000000000009