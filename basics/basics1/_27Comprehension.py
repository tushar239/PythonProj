# https://www.w3schools.com/python/python_lists_comprehension.asp
# https://pythonguides.com/python-list-comprehension-using-if-else/
"""
Comprehension:
Shorter way to write the code with loops and if conditions in just one line.

List comprehension:
It offers a shorter syntax when you want to create a new list based on the values of an existing list.
"""

# Example:
# Based on a list of fruits, you want a new list, containing only the fruits with the letter "a" in the name.

# Without list comprehension you will have to write a for statement with a conditional test inside:
fruits = ["apple", "banana", "cherry", "kiwi", "mango"]
newlist = []

for x in fruits:
    if "a" in x:  # implicitly calls a magic method __contains__ of str class
        newlist.append(x)

print(newlist)  # ['apple', 'banana', 'mango']

# With list comprehension you can do all that with only one line of code:
# Syntax: result = expression for item in iterable if condition == True
is_nice = True
state = "not nice" if not is_nice else "nice"
print(state)  # nice

# Syntax: newlist = [expression for item in iterable if condition == True]
fruits = ["apple", "banana", "cherry", "kiwi", "mango", "apple"]
newlist = [x for x in fruits if "a" in x]  # for loop and if conditions are merged in one line
print(newlist)  # ['apple', 'banana', 'mango', "apple"]

# Creating a Set
newlist = {x for x in fruits if "a" in x}
print(newlist)  # {'mango', 'apple', 'banana'}

fruits = ["apple", "banana", "cherry", "kiwi", "mango"]

newlist = [x for x in fruits if "a" not in x]
print(newlist)  # ['cherry', 'kiwi']

newlist = [x for x in fruits if x != "apple"]  # Only accept items that are not "apple"
print(newlist)  # ['banana', 'cherry', 'kiwi', 'mango']

newlist = [x for x in fruits]  # With no if statement
print(newlist)  # ['apple', 'banana', 'cherry', 'kiwi', 'mango']

# The expression is the current item in the iteration, but it is also the outcome, which you can manipulate before it ends up like a list item in the new list.
newlist = [x.upper() for x in fruits]
print(newlist)  # ['APPLE', 'BANANA', 'CHERRY', 'KIWI', 'MANGO']

# nested for loops and if loop
newlist = [x.upper() for i in range(len(fruits)) for x in fruits[i] if 'a' in fruits[i]]
print(newlist)  # ['A', 'P', 'P', 'L', 'E', 'B', 'A', 'N', 'A', 'N', 'A', 'M', 'A', 'N', 'G', 'O']

# Set all values in the new list to 'hello'
newlist = ['hello' for x in fruits]
print(newlist)  # ['hello', 'hello', 'hello', 'hello', 'hello']

# Return "orange" instead of "banana"
# when you need else condtion, for loop goes after if-else
newlist = [x if x != "banana" else "orange" for x in fruits]
print(newlist)  # ['apple', 'orange', 'cherry', 'kiwi', 'mango']

# Dictionary Comprehensions
# https://www.geeksforgeeks.org/comprehensions-in-python/
input_list = [1, 2, 3, 4, 5, 6, 7]
dict_using_comp = {var: var ** 3 for var in input_list if var % 2 != 0}
print("Output Dictionary using dictionary comprehensions:", dict_using_comp)

# Set Comprehension
input_list = [1, 2, 3, 4, 4, 5, 6, 6, 6, 7, 7]
set_using_comp = {var for var in input_list if var % 2 == 0}
print("Output Set using set comprehensions:", set_using_comp)

###########################################################
