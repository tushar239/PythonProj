import numpy as np

a = np.matrix("1,2,3,4;4,5,6,7;7,8,9,10")
print(a)
'''
[[ 1  2  3  4]
 [ 4  5  6  7]
 [ 7  8  9 10]]
'''

a[1,1] = 25
# a[1][2] = 27  # syntax not allowed
print(a)
'''
[[ 1  2  3  4]
 [ 4 25  6  7]
 [ 7  8  9 10]]
'''