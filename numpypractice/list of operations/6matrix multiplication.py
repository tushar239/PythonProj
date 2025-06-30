import numpy as np

a = np.array([[1, 2],
              [3, 4]])
b = np.array([[11, 22],
              [33, 44]])

print(np.dot(a, b))
print(a @ b)
'''
[[ 77 110]
 [165 242]]
'''
