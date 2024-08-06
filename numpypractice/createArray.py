import numpy as np
from inheritance.Child import Child

'''
numpy is is used to create n-dimensional array and so it is called nd array

creates a numpy array
'''
array = np.array([2, 3, 4, 5])  # creating a numpy array from a list
print(array)  # [2 3 4 5]
print(type(array))  # <class 'numpy.ndarray'> type is a class, and you are creating an object of that class and trying to print it.

array = np.array((2, 3, 4, 5))  # creating a numpy array from a tuple
print(array)  # [2 3 4 5]
print(array[0])  # 2

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
print(array)  # ['2' '3' 'hi' '5']

array = np.array([2, 3, "hi", Child()])  # ['2' '3' 'hi']  Child instance is going to be garbage collected

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
print(array)  # (array([1. , 2.8, 4.6, 6.4, 8.2]), 1.8)

'''
returns equally spaced numbers with in the given range based on step size

start - start o finterval range
stop - end of interval range (excluding stop number)
step - step size of interval
'''
array = np.arange(start=1, stop=5, step=.5)
print(array)  # [1.  1.5     2.  2.5     3.  3.5     4.  4.5]

'''
return an array of given shape and type filled with ones
shape - integer or sequence of integers
        shape determines number of rows and columns
dtype = data type(default:float)
'''
# it returns 2D array of 5 rows and 4 cols, 1s are filled up as ints
array = np.ones((5, 4), int)
print(array)
'''
[[1 1 1 1]
 [1 1 1 1]
 [1 1 1 1]
 [1 1 1 1]
 [1 1 1 1]]
'''

# it returns 3D array of 5 rows, 4 cols , in that each cell has 2 elements. 1s are filled up as ints
array = np.ones((5, 4, 2), int)
print(array)
'''
[[[1 1]
  [1 1]
  [1 1]
  [1 1]]

 [[1 1]
  [1 1]
  [1 1]
  [1 1]]

 [[1 1]
  [1 1]
  [1 1]
  [1 1]]

 [[1 1]
  [1 1]
  [1 1]
  [1 1]]

 [[1 1]
  [1 1]
  [1 1]
  [1 1]]]
'''

'''
returns an array of given shape and type filled with zeros
shape - integer or sequence of integers
        shape determines number of rows and columns
dtype = data type(default:float)
'''
array = np.zeros((5, 4))
print(array)
'''
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
'''

'''
returns an array of given shape filled with random values
'''
array = np.random.rand(3)
print(array)  # [0.4565651  0.7101034  0.50924081]

# generates 2D array of 5 rows and 3 cols filled with random numbers
array = np.random.rand(5, 3)
print(array)
'''
[[0.65511054 0.78611646 0.10467883]
 [0.62518694 0.1005175  0.96188029]
 [0.25696408 0.01781009 0.53449356]
 [0.35996827 0.29416267 0.61813613]
 [0.82316784 0.86706935 0.90414276]]
'''

# generates 3D array of 5 rows and 3 cols and each cell having 2 random numbers
array = np.random.rand(5, 3, 2)
print(array)
'''
[[[0.70266655 0.6811018 ]
  [0.21893095 0.1254682 ]
  [0.4469065  0.07394416]]

 [[0.29631807 0.15017304]
  [0.64406398 0.78102127]
  [0.96732061 0.30657564]]

 [[0.44664486 0.05322068]
  [0.1761625  0.61774614]
  [0.2437411  0.14295917]]

 [[0.7748825  0.85084441]
  [0.99227456 0.28233192]
  [0.30785338 0.60687447]]

 [[0.50641863 0.204215  ]
  [0.45005414 0.42171476]
  [0.68092093 0.35879159]]]
'''

array = np.random.randint(low=0, high=3, size=5)
print(array)  # [0 2 0 2 0] random numbers are generated in between 0 to 2. size of the array will be 5

array = np.random.randint(low=-3, high=10, size=(2, 3))
print(array)
'''
[[-3  6 -1]
 [ 9  0 -2]]
'''

'''
returns equally spaced numbers based on log scale
start - start value of the sequence
stop - end value of the sequence
num - number of samples to generate (default : 50)
endpoint - if true, stop is the last sample
base - base of the log space (default : 10.0)
dtype = type of output array
'''
array = np.logspace(start=1, stop=10, num=5, endpoint=True, base=10.0, dtype=float)
print(array)  # [1.00000000e+01 1.77827941e+03 3.16227766e+05 5.62341325e+07 1.00000000e+10]

# random samples of one dimensional array.
# total 15 samples in between 0 and 9
array = np.random.choice(10, 15)
print(array)  # [5 8 9 1 6 7 9 3 7 7 3 6 9 2 9]


