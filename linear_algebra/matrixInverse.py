# https://www.youtube.com/watch?v=kWorj5BBy9k
# https://www.youtube.com/watch?v=HYWeEx21WWw
# https://www.youtube.com/watch?v=xfhzwNkMNg4

'''

inverse of matrix A = A^-1 = 1/|A| x Adjoin(A)
|A| means determinant of A
Singular matrix means the determinant of a matrix is 0.
You can’t find an inverse of a matrix, if the determinant of a matrix is 0.
'''

import numpy as np
# linalg is a package inside numpy package
import numpy.linalg as la

a = np.matrix("4,5,16,7;2,-3,2,3;3,4,5,6;4,7,8,9")
print(a)
'''
[[ 4  5 16  7]
 [ 2 -3  2  3]
 [ 3  4  5  6]
 [ 4  7  8  9]]
'''
result = la.inv(a)
print(result)

'''
[[ 9.37500000e-02 -4.68750000e-01  3.68750000e+00 -2.37500000e+00]
 [ 1.00929366e-16 -2.50000000e-01  5.00000000e-01 -2.50000000e-01]
 [ 9.37500000e-02  3.12500000e-02 -3.12500000e-01  1.25000000e-01]
 [-1.25000000e-01  3.75000000e-01 -1.75000000e+00  1.25000000e+00]]
'''


a = np.matrix("2,1,2;1,0,1;3,1,3")
print(a)
'''
[[2 1 2]
 [1 0 1]
 [3 1 3]]
'''
determinant = la.det(a)
print(determinant) # 0.0
# As determinant of this matrix is 0, it is called a Singular Matrix. Its inverse can't be determined.

# result = la.inv(a) # numpy.linalg.LinAlgError: Singular matrix
# print(result)