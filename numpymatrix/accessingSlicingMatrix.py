import numpy as np

a = np.matrix("1,2,3,4;4,5,6,7;7,8,9,10")
print(a)
'''
[[ 1  2  3  4]
 [ 4  5  6  7]
 [ 7  8  9 10]]
'''

# a[start row:stop row:step, start col:stop col:step]
subMatrix1 = a[1:3, 1:3]
print(subMatrix1)
'''
[[5 6]
 [8 9]]
'''

subMatrix1 = a[1]
print(subMatrix1)
'''
[[4 5 6 7]]
'''

subMatrix = a[1:3]
print(subMatrix)

'''
[[ 4  5  6  7]
 [ 7  8  9 10]]
'''

# IMP: Modifying subMatrix will modify an original matrix also
subMatrix[0,0] = 31
print(subMatrix)
'''
[[31  5  6  7]
 [ 7  8  9 10]]
'''
print(a)
'''
[[ 1  2  3  4]
 [31  5  6  7]
 [ 7  8  9 10]]
'''