import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

list = [[0.08581368, 1.0],
        [0.09701243, 1.0],
        [0.09433366, 1.0],
        [0.09156187, 1.0],
        [0.07984225, 1.0],
        [0.0643277, 1.0],
        [0.0585423, 1.0],
        [0.06568569, 1.0],
        [0.06109085, 1.0],
        [0.06639259, 1.0],
        [0.0614257, 1.0],
        [0.07474514, 1.0],
        [0.02797827, 1.0],
        [0.02379269, 1.0],
        [0.02409033, 1.0]]
sliced_list = list[0:10][0][0] # # 0 to 9 rows, 0th element in that([0.08581368, 1.0]) and 0th element in that (0.08581368)
print(sliced_list) # 0.08581368

# sliced_list = list[0:10, 1] # you can't do this in list. you can do this in numpy array.

aa = np.array(list)

c = aa[0:10][0][0] # 0 to 9 rows, 0th element in that([0.08581368, 1.0]) and 0th element in that (0.08581368)
print(c) # 0.08581368

a = aa[0:10, 0] # 0 to 9 rows and 0th column in them
print(a) # [0.08581368 0.09701243 0.09433366 0.09156187 0.07984225 0.0643277 0.0585423  0.06568569 0.06109085 0.06639259]

b = aa[0:10, 1] # 0 to 9 rows and 1st column in them
print(b) # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
