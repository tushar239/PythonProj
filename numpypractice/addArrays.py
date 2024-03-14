import numpy as np

a = np.array([[1,2,3], [4,5,6], [7,8,9]])

b = np.arange(11, 20, 1).reshape((3,3))

# it will do sum of elements and create a new array
result = np.add(a, b)
print(result)

'''
[[12 14 16]
 [18 20 22]
 [24 26 28]]
'''
