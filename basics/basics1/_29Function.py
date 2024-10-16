"""
Function -it is something to which you give some input and it gives some output.
Many a times, code that is being used repeatedly (more than once) is kept inside the function to reduce repetition
and so for maintaining better readability and maintainability
"""

# example of one function's output becomes another function's input
len(list(range(1, 7)))

# look at FindBiggestNumber.py
# This kind of code can be used at multiple places to find out the biggest number.
# so, it is better to put it in a function and call that function wherever you need.

# mylist = [1, 3, 6, 7, 4, 5]
# or
mylist = list()
mylist.extend([1, 3, 6, 7, 4, 5])
print(mylist)
# print(type(mylist))
print(max(mylist))  # 7
print(max(4, 7, 9, 1, 5))  # 9
print(min(4, 7, 9, 1, 5))  # 1
