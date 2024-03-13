# https://www.geeksforgeeks.org/python-lists-vs-numpy-arrays/


import sys
import numpy as np

array = [11, 12, 13, 14, 15]
size = sys.getsizeof(array[0]) * len(array)
print(size)  # 140 bytes

numpyArray = np.array([11, 12, 13, 14, 15])
size = numpyArray.itemsize * numpyArray.size
print(size)  # 20 bytes
