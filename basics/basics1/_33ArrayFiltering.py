import numpy as np

l = [41, 42, 43, 44, 45, 46, 47]  # list
a = np.array(l)

# Array Filtering, Creating a View from an array
b = a[2:4]
print("SubArray: ", b)  # [43, 44]  --- subarray is just a view of original array. It points to the same memory location. So, if you change anything in subarray, that change will be reflected in original array as well.
b[:] = 111  # Assign value to all the elements in b
print("SubArray: ", b)  # [111 111]
print("array: ", a)  # [ 41  42 111 111  45  46  47]

print()

# Array Filtering
# Create an array from the elements on index 0 and 2
d = np.array([41, 42, 43, 44])
x = [True, False, True, False]
newarr = d[x]  # This is not a view of original array d. So, change in newarr will not change anything in original array d.
print("Filtered Array: ", newarr)
newarr[0] = 91
print("Filtered Array after change: ", newarr)
print("Original Array: ", d)

print()

# https://www.w3schools.com/python/numpy/numpy_array_filter.aspv

arr = np.array([41, 42, 43, 44])

# Create an empty list
filter_arr = []

# go through each element in arr
for element in arr:
  # if the element is higher than 42, set the value to True, otherwise False:
  if element > 42:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr[filter_arr]

print(filter_arr)  # [False, False, True, True]
print(newarr)  # [43 44]
