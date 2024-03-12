'''
Numpy supports vectorized operations.

Numpy arrays are homogeneous in nature means it is an array that contains data of a single type only.
Python’s lists and tuples are unrestricted in the type of data they contain.
The concept of vectorized operations on NumPy allows the use of more optimal and pre-compiled functions and mathematical operations on NumPy array objects and data sequences.
The Output and Operations will speed up when compared to simple non-vectorized operations.

Array operations are carried out in C and hence the universal functions in numpy are faster than
operations carried out on python lists.
'''
import timeit

import numpy as np
import time

# Just like time module, there is timeit module also
# https://stackoverflow.com/questions/14452145/how-to-measure-time-taken-between-lines-of-code-in-python

# vectorized sum
start = time.time_ns()
sum = np.sum(np.arange(15000))
end = time.time_ns()
print("time taken to sum by numpy in nano seconds: " + str(end - start))

# iterative sum
start = time.time_ns()
sum = 0
for item in range(0, 15000):
    sum += item
end = time.time_ns()
print("time taken to sum by iterative sum in nano seconds: " + str(end - start))

'''
timeit module

https://www.geeksforgeeks.org/timeit-python-examples/

Syntax: timeit.timeit(stmt, setup, timer, number

stmt: which is the statement you want to measure; it defaults to ‘pass’.
setup: which is the code that you run before running the stmt; it defaults to ‘pass’. 
We generally use this to import the required modules for our code.

timer: which is a timeit.Timer object; it usually has a sensible default value so you don’t have to worry about it.

number: which is the number of executions you’d like to run the stmt. 


Returns the number of seconds it took to execute the code.


Well, how about using a simple time module? Just save the time before and after the execution of code and subtract 
them! But this method is not precise as there might be a background process momentarily running which disrupts the 
code execution and you will get significant variations in the running time of small code snippets. Timeit runs your 
snippet of code millions of times (default value is 1000000) so that you get the statistically most relevant 
measurement of code execution time!'''

# code snippet to be executed only once
mysetup = "array = range(15000)"

# code snippet whose execution time is to be measured
mycode = ''' 
def example():
    sum = sum(array)
'''

# timeit statement
print(timeit.timeit(setup=mysetup,
                    stmt=mycode,
                    number=1000)) # stmt will be run 1000 time and average of it will be taken
# another way
print(timeit.timeit(stmt='sum(array)',
                    setup='array = range(15000)',
                    number=1000)) # stmt will be run 1000 time and average of it will be taken
