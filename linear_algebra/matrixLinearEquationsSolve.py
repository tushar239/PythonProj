import numpy as np
# linalg is a package inside numpy package
import numpy.linalg as la

'''
Linear equations:

3x + y + 2z = 2
3x + 2y +5z = -1
6x + 7y +8z = 3

Find out x, y and z values

Create matrix (A) from coefficients 

[[3,1,2],
 [3,2,5],
 [6,7,8]]
 
Create a matrix (B)

[x
 y 
 z]
 
create a matrix (C)

[2
 -1 
 3]
 
A * B = C    --- see how matrices are multiplied in addSubtractMultiplyDivideTwoMatrices.py

You need to find out the result of B

B = A^-1 * C    --- see how the inverse of a matrix is found in matrixInverse.py

This can be easily solved using solve(A, C) method
'''
A = np.array(
    [[3,1,2],
    [3,2,5],
    [6,7,8]])
# B = np.array(['x','y','z']).transpose()
C = np.array([2, -1, 3]).transpose()
result = la.solve(A, C)
print(result)  # values of x,y and z are [ 1.24242424  0.81818182 -1.27272727]

