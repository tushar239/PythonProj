'''
np.unique(a)              # unique values
np.sort(a)                # sort array
np.argsort(a)             # indices of sorted values
np.clip(a, 0, 5)          # limit values between 0 and 5
np.isnan(a), np.isinf(a)  # check for NaN or Inf
'''
import math

import numpy as np

a = np.array([1,2,1,3,2])
print(np.unique(a)) # [1 2 3]

print(math.isinf(float('inf')))  # Output: True
print(math.isinf(-float('inf')))  # Output: True
print(math.isinf(10.0))  # Output: False

a = np.array([1,2,np.nan,3,2])
print(np.isnan(a)) # [False False  True False False]


