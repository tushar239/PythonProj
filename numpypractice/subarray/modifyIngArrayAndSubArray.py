import numpy as np

a = np.array([[1,2,3], [4,5,6], [7,8,9]])

# Modifying an element
a[2][2] = 19  # you can use a[2,2]=19 also
print(a)

'''
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8 19]]
'''

subArray = a[1:3, 0:2]
print(subArray)
'''
[[4 5]
 [7 8]]
'''

# IMPORTANT - Modifying a subArray will modify the original array as well
subArray[0][0] = 14
print(subArray)
print(a)
'''
[[14  5]
 [ 7  8]]

[[ 1  2  3]
 [14  5  6]
 [ 7  8  9]]
'''