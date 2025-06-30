import numpy as np

a = np.array([1, 2, 3])
b = np.array([2, 2, 2])

print(a > b)       # [False, False, True]
print(np.where(a > b, 1, 0))   # condition-based selection [0 0 1]
print(np.any(a > 2))          # True
print(np.all(a > 0))           # True
