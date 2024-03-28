import numpy as np

a = np.array([[1,2,3], [4,5,6], [7,8,9]])
b = np.arange(11, 20, 1).reshape((3,3))
print(a)
print(b)

'''
[[1 2 3]
 [4 5 6]
 [7 8 9]]
 
[[11 12 13]
 [14 15 16]
 [17 18 19]]
'''

'''
matrix3 = numpy.add(matrix1, matrix2)
It does elementwise addition between two matrices
'''
# it will do sum of elements and create a new matrix
result = np.add(a, b)
print(result)

'''
[[12 14 16]
 [18 20 22]
 [24 26 28]]
'''

'''
matrix3 = numpy.subtract(matrix1, matrix2)
It does elementwise subtraction between two matrices
'''
result = np.subtract(b, a)
print(result)
'''
[[10 10 10]
 [10 10 10]
 [10 10 10]]
'''

'''
matrix3 = numpy.dot(matrix1, matrix2)
OR
matrix3 = numpy.matmult(matrix1, matrix2)

It does elementwise multiplication between two matrices

It multiplies rows and columns
Number of rows in matrix1 should be same as number of columns in matrix2
'''

result = np.dot(a, b) # gives the same result as np.matmult(a, b)
print(result)


'''
It multiplies rows and columns
IMPORTANT: Understand how two matrices are multiplied. This concept will be used throughout in Linear Algebra section.

Number of rows in matrix1 should be same as number of columns in matrix2

[[1*11 + 2*14 + 3*17        1*12 + 2*15 + 3*18      1*13 + 2*16 + 3*19]
 [4*11 + 5*14 + 6*17        4*12 + 5*15 + 6*18      4*13 + 5*16 + 6*19]
 [7*11 + 8*14 + 9*17        7*12 + 8*15 + 9*18      7*13 + 8*16 + 9*19]]

[[ 90  96 102]
 [216 231 246]
 [342 366 390]]
'''

result = np.matmul(a, b)  # gives the same result as np.dot(a, b)
print(result)
'''
[[ 90  96 102]
 [216 231 246]
 [342 366 390]]
'''


'''
matrix3 = numpy.multiply(matrix1, matrix2)
It performs elementwise multiplication between two matrices
'''
result = np.multiply(a, b)
print(result)
'''
[[ 11  24  39]
 [ 56  75  96]
 [119 144 171]]
'''

'''
matrix3 = numpy.divide(matrix1, matrix2)
It performs elementwise division between two matrices
'''
result = np.divide(a, b)
print(result)
'''
[[0.09090909 0.16666667 0.23076923]
 [0.28571429 0.33333333 0.375     ]
 [0.41176471 0.44444444 0.47368421]]
'''