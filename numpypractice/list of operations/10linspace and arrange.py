import numpy as np

np.linspace(0, 10, 5)   # [0, 2.5, 5.0, 7.5, 10.0]

# np.arange(start, end-exclusive, step)
a = np.arange(0, 10, 2)     # [0, 2, 4, 6, 8]

'''
range vs np.arange

- range() is a built-in Python function that returns an immutable sequence of numbers, specifically a range object.
- numpy.arange() is a function from the NumPy library that returns a NumPy ndarray (N-dimensional array).

- range() strictly works with integers.
- numpy.arange() can handle floating-point numbers as arguments and can generate arrays containing floats.

- range() is more memory-efficient for large sequences because it generates numbers on the fly (it's an iterator) rather than storing all numbers in memory at once.
- numpy.arange() creates and stores the entire array in memory, which can be less efficient for extremely large sequences if memory is a concern.

- range() is primarily used for iteration in for loops and for generating sequences of integers.
- numpy.arange() is designed for numerical computations and array manipulation, offering performance benefits for mathematical operations on arrays.

'''

a = np.arange(0, 10, 2.5)
print(a) # [0.  2.5 5.  7.5]

a = range(0, 10, 2)
print(a) # range(0, 10, 2)
