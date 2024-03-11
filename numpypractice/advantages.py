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
import timeit
'''
# vectorized sum
print(np.sum(np.arange(15000)))

print("Time taken by vectorized sum : ", end="")
print(timeit.timeit(np.sum(np.arange(15000))))

# iterative sum
total = 0
for item in range(0, 15000):
    total += item
a=total
print("\n" + str(total))

print("Time taken by iterative sum : ", end="")
print(timeit.timeit(a))
'''
# code snippet to be executed only once
mysetup = "5+5"

print(timeit.timeit(mysetup))