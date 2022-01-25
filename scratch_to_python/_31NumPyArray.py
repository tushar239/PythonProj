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

a = np.array(l)
print(type(a))  # <class 'numpy.ndarray'>
print(a)  # ['5' '7' '9' '1' 'abc']
a.put([1, 2], ["hi", "bi"])  # ['5' 'hi' 'bi' '1' 'abc']
print("numpy array: ", a)
print("sorted numpy array: ", np.sort(a, kind='mergesort'))  # ['1' '5' 'abc' 'bi' 'hi']

b = a[2:4]
print("SubArray: ", b)  # ['bi' '1']  --- subarray is just a view of original array. It points to the same memory location. So, if you change anything in subarray, that change will be reflected in original array as well.
b[:] = 111  # Assign value to all the elements in b
print("SubArray: ", b)  # ['111' '111']
print("array: ", a)  # ['5' 'hi' '111' '111' 'abc']

c = np.copy(a)  # this will create a new array pointing to different memory location
print("Copied array: ", c)  # ['5' 'hi' '111' '111' 'abc']

ll = [1, 2, 5, 9]
aa = ar.array("i", ll)
print(type(aa))  # <class 'array.array'>
print(aa)  # array('i', [1, 2, 5, 9])
print(aa.pop(1))  # 2




