'''
'range' is a class, not a function in builtins.py

It returns an object that produces a sequence of integers from start (inclusive)
to stop (exclusive) by step.  range(i, j) produces i, i+1, i+2, ..., j-1.

The advantage of the range type over a regular list or tuple is that a range object will always take the same (small) amount of memory, 
no matter the size of the range it represents (as it only stores the start, stop and step values, calculating individual items and subranges as needed).
'''

num_range = range(0, 15, 2)  # start=0, stop=5, step=1
print(type(num_range)) # <class 'range'>
print(num_range)  # range(0, 15, 2)
print('Values = ', num_range[0], num_range[1], num_range[2], num_range[3], num_range[4])  # Values =  0 2 4 6 8


for i in num_range:
    print(i)

'''
The range object can be converted to the other iterable types such as list, tuple, and set.
'''
print(list(range(5)))  # [0, 1, 2, 3, 4]
print(tuple(range(5)))  # (0, 1, 2, 3, 4)
print(set(range(5)))  # {0, 1, 2, 3, 4}