import numpy as np
# linalg is a package inside numpy package
import numpy.linalg as la

'''
Matrix Rank
https://www.youtube.com/watch?v=zksRGHYD76g
'''


a = np.matrix([[1, 2],
               [4, 5]])
result = la.matrix_rank(a)
print(result) # 2

a = np.matrix([[1, 2],
               [4, 5],
               [6, 7]])
result = la.matrix_rank(a)
print(result) # 2


# creating 4 X 4 matrix
a = np.matrix("4,5,16,7;2,-3,2,3;3,4,5,6;4,7,8,9")
print(a)
'''
[[ 4  5 16  7]
 [ 2 -3  2  3]
 [ 3  4  5  6]
 [ 4  7  8  9]]
'''
result = la.matrix_rank(a)
print(result) # 4