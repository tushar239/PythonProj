import numpy as np

array = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(array.shape)  # returns a tuple with number of rows and cols - (3, 3)

# array.shape returns a tuple. So, you can access a tuple to know number of rows and cols
print(array.shape[0])  # gives number of rows - 3
print(array.shape[1])  # gives number of cols - 3
