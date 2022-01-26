"""
Python has its own package called 'array'.
NumPy array gives better features, so normally developers use NumPy array.

Difference between list and array:
list can have elements of different data types, array can't
If you use Python's array with different data types, then it will give error.
NumPy's array will convert all different typed elements to string

"""

import numpy as np
import array as ar

l = [5, 7, 9, 1, "abc"]
print(type(l)) # <class 'list'>

# Array Creation
a = np.array(l)
print(type(a))  # <class 'numpy.ndarray'>
print(a)  # ['5' '7' '9' '1' 'abc']
a.put([1, 2], ["hi", "bi"])  # ['5' 'hi' 'bi' '1' 'abc']
print("numpy array: ", a)
print("sorted numpy array: ", np.sort(a, kind='mergesort'))  # ['1' '5' 'abc' 'bi' 'hi']

# Copying an array
c = np.copy(a)  # this will create a new array pointing to different memory location
print("Copied array: ", c)  # ['5' 'hi' '111' '111' 'abc']

# Using Python's inbuilt Array
ll = [1, 2, 5, 9]
aa = ar.array("i", ll)
print(type(aa))  # <class 'array.array'>
print(aa)  # array('i', [1, 2, 5, 9])
print(aa.pop(1))  # 2




