import numpy as np

arr = np.array([1, 5, 2, 8, 3])
max_index = np.argmax(arr)
print(f"Index of maximum value: {max_index}")  # Output: 3


arr_2d = np.array([[10, 17, 25],
                   [15, 11, 22]])

# Max index along axis 0 (columns)
max_index_col = np.argmax(arr_2d, axis=0)
print(f"Indices of max along columns: {max_index_col}")  # Output: [1 0 0]

# Max index along axis 1 (rows)
max_index_row = np.argmax(arr_2d, axis=1)
print(f"Indices of max along rows: {max_index_row}")  # Output: [2 2]