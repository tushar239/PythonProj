a = np.array([1, 2, 3])
b = np.array([2, 2, 2])

a > b        # [False, False, True]
np.where(a > b, 1, 0)   # condition-based selection
np.any(a > 2)           # True
np.all(a > 0)           # True
