import numpy as np

a = np.arange(11, 20, 1).reshape(3, 3)
b = np.array([[1,2,3], [4,5,6], [7,8,9]])

print(a)
print(b)

'''
[[11 12 13]
 [14 15 16]
 [17 18 19]]
 
[[1 2 3]
 [4 5 6]
 [7 8 9]]
'''

result = np.multiply(a, b)
print(result)

'''
[[ 11  24  39]
 [ 56  75  96]
 [119 144 171]]
'''