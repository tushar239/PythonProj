'''
Numpy supports vectorized operations.

Numpy arrays are homogeneous in nature means it is an array that contains data of a single type only.
Python’s lists and tuples are unrestricted in the type of data they contain.
The concept of vectorized operations on NumPy allows the use of more optimal and pre-compiled functions and mathematical operations on NumPy array objects and data sequences.
The Output and Operations will speed up when compared to simple non-vectorized operations.

Array operations are carried out in C and hence the universal functions in numpy are faster than
operations carried out on python lists.
'''

import numpy as np
import time
# Just like time module, there is timeit module also
# https://stackoverflow.com/questions/14452145/how-to-measure-time-taken-between-lines-of-code-in-python

# vectorized sum
start = time.time_ns()
sum = np.sum(np.arange(15000))
end = time.time_ns()
print("time taken to sum by numpy in nano seconds: " +str(end-start))


# iterative sum
start = time.time_ns()
sum = 0
for item in range(0, 15000):
    sum += item
end = time.time_ns()
print("time taken to sum by iterative sum in nano seconds: " +str(end-start))