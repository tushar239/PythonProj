import numpy as np

'''
recasts an array to new shape without changing its data
Here, arange(sart,stop,step) will return 1D array.
If you want to reshape it into nD array, then you can use reshape.
'''
array = np.arange(1, 10, 1).reshape((3,3))
print(array)

'''
[[1 2 3]
 [4 5 6]
 [7 8 9]]
'''

# array = np.arange(1, 11, 1).reshape((3,3)) # cannot reshape array of size 10 into shape (3,3)


