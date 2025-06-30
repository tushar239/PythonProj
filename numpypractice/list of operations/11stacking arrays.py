import numpy as np

a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5, 6]])

c = np.vstack((a, b))   # vertical stack
d = np.hstack((a, a))   # horizontal stack

print(c)
'''
[[1 2]
 [3 4]
 [5 6]]
'''
print(d)
'''
[[1 2 1 2]
 [3 4 3 4]]
'''