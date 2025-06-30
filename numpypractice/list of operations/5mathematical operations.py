'''
a + b
a - b
a * b        # element-wise multiplication
a / b
a ** 2       # power
'''

import numpy as np

a = np.array([[1, 2],
              [3, 4]])
b = np.array([[11, 22],
              [33, 44]])
print(a+b)
'''
[[12 24]
 [36 48]]
'''
print(a*b)
'''
[[ 11  44]
 [ 99 176]]
'''
