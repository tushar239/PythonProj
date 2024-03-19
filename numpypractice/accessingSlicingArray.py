import numpy as np

a = np.array([[1,2,3], [4,5,6], [7,8,9]])

# Both syntaxes work to access the elements from 2D array
print(a[0][1])
print(a[0,1])


print(a[2][2])
print(a[2,2])

'''
slicing (extracting) - It is just like slicing a python list
syntax for 1D array - array[start : stop : step]
syntax for 2D array - array[row start : row stop : row step, col start : col stop : col step]

default start = 0
default stop = last index + 1
default step = 1
'''

a = np.array([[1,2,3], [4,5,6], [7,8,9]])
# retrieving sub-array
subArray = a[1:3] # default step=1
print(subArray)
'''
[[4 5 6]
 [7 8 9]]
'''


subArray = a[::2]  # default start=0, stop=3
print(subArray)
'''
[[1 2 3]
 [7 8 9]]
'''

# slicing (extracting column from extracted rows)
subArray = a[1:3, 0]
print(subArray) # [4 7]

subArray = a[1:3, 0:2]
print(subArray)
'''
[[4 5]
 [7 8]]
'''