# initializes all the 10 spaces with 0’s
a = [0] * 10
print("Initialising empty list with zeros: ", a)

# initializes all the 10 spaces with None
b = [None] * 10
print("Initialising empty list of None: ", b)

# initializes a 4 by 3 array matrix all with 0's
c =  [[0] * 4] * 3
print("Initialising 2D empty list of zeros: ", c)

# empty list which is not null, it's just empty.
d = []
print("Initialising empty list of zeros: ", d)

'''
Python List vs Numpy Array
https://www.geeksforgeeks.org/python-lists-vs-numpy-arrays/

Python List is like a LinkedList.
Numpy array ls like an ArrayList

Element Overhead: Lists in Python store additional information about each element, such as its type and reference count. This overhead can be significant when dealing with a large number of elements.
Datatype: Lists can hold different data types, but this can decrease memory efficiency and slow numerical operations.
Memory Fragmentation: Lists may not store elements in contiguous memory locations, causing memory fragmentation and inefficiency.
Performance: Lists are not optimized for numerical computations and may have slower mathematical operations due to Python’s interpretation overhead. They are generally used as general-purpose data structures.
Functionality: Lists can store any data type, but lack specialized NumPy functions for numerical operations.
'''