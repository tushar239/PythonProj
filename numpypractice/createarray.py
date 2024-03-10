import numpy as np
from inheritance.Child import Child

'''
numpy is is used to create n-dimensional array and so it is called nd array

creates a numpy array
'''
array = np.array([2, 3, 4, 5])  # creating a numpy array from a list
print(array)  # [2 3 4 5]
print(type(array)) # <class 'numpy.ndarray'> type is a class and you are creating an object of that class and trying
# to print it.

array = np.array((2, 3, 4, 5))  # creating a numpy array from a tuple
print(array)  # [2 3 4 5]
print(array[0]) # 2

# trying to use a dictionary to create a numpy array, but doesn't work properly
'''
array = np.array({"1":"hi1", "2":"hi2", "3":"hi3"})
print(array) # {1: 'hi1', 2: 'hi2', 3: 'hi3'}
print(type(array)) # <class 'numpy.ndarray'>
print(array["1"]) # IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
'''

# Numpy can handle different categorical entities
# If one element is a string, it converts all others to string also
array = np.array([2, 3, "hi", 5])
print(array) # ['2' '3' 'hi' '5']

array = np.array([2, 3, "hi", Child()]) # ['2' '3' 'hi' '5']  Child instance is going to be garbage collected

'''
creates an array of equally spaced numbers within the given range based on the sample number
start = starting number
stop = end number
num = number of elements
endpoint = include end number in the array or not
dtype = data type of elements (default is float)
retstep = It's an increment value. Include step that was used to generate the elements of an array
'''
array = np.linspace(start=1, stop=10, num=5, endpoint=False, dtype=float, retstep=True)
print(array) # (array([1. , 2.8, 4.6, 6.4, 8.2]), 1.8)

'''

'''
array = np.arange(start=1, stop=5, step=.5)
print(array) # [1.  1.5     2.  2.5     3.  3.5     4.  4.5]