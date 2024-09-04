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

################  Creating an empty list #########################
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
###################### Creating a list with items ###############################

list1 = ["apple", "banana", "cherry"]
list2 = [1, 5, 7, 9, 3]
list3 = [True, False, False]

# A list with strings, integers and boolean values:
list1 = ["abc", 34, True, 40, "male"]
#allows duplicates
thislist = ["apple", "banana", "cherry", "apple", "cherry"]

# creating a list with list() constructor
thislist = list(("apple", "banana", "cherry")) # note the double round-brackets

# length of the list
print(len(thislist))
######################### accessing the list item ###################
# accessing the list
print(thislist[1])

# negative indexing
print(thislist[-1])

# The search will start at index 2 (included) and end at index 5 (not included).
print(thislist[2:5])
print(thislist[:4])
print(thislist[2:])
print(thislist[-4:-1])

# check if item exists
if "apple" in thislist:
  print("Yes, 'apple' is in the fruits list")

################## Changing the item ######################
thislist = ["apple", "banana", "cherry"]
thislist[1] = "blackcurrant"
print(thislist) # ['apple', 'blackcurrant', 'cherry']

thislist = ["apple", "banana", "cherry", "orange", "kiwi", "mango"]
thislist[1:3] = ["blackcurrant", "watermelon"]
print(thislist) # ['apple', 'blackcurrant', 'watermelon', 'orange', 'kiwi', 'mango']

thislist = ["apple", "banana", "cherry"]
thislist[1:2] = ["blackcurrant", "watermelon"]
print(thislist) # ['apple', 'blackcurrant', 'watermelon', 'cherry']

thislist = ["apple", "banana", "cherry"]
thislist[1:3] = ["watermelon"]
print(thislist) # ['apple', 'watermelon']

########## insert an item ##########
thislist = ["apple", "banana", "cherry"]
thislist.insert(2, "watermelon")
print(thislist) # ['apple', 'banana', 'watermelon', 'cherry']

thislist = ["apple", "banana", "cherry"]
thislist.append("orange")
print(thislist) # ['apple', 'banana', 'cherry', 'orange']

thislist = ["apple", "banana", "cherry"]
tropical = ["mango", "pineapple", "papaya"]
thislist.extend(tropical)
print(thislist) # ['apple', 'banana', 'cherry', 'mango', 'pineapple', 'papaya']

# Adding a tuple to the list
thislist = ["apple", "banana", "cherry"]
thistuple = ("kiwi", "orange")
thislist.extend(thistuple)
print(thislist) # ['apple', 'banana', 'cherry', 'kiwi', 'orange']

############ Remove the item from the list ###################
thislist = ["apple", "banana", "cherry"]
thislist.remove("banana")
print(thislist) # ['apple', 'cherry']

thislist = ["apple", "banana", "cherry", "banana", "kiwi"]
thislist.remove("banana") # Remove the first occurrence of "banana"
print(thislist) # ['apple', 'cherry', 'banana', 'kiwi']

thislist = ["apple", "banana", "cherry"]
thislist.pop(1) # Remove by index number
print(thislist) # ['apple', 'cherry']

thislist = ["apple", "banana", "cherry"]
thislist.pop() # Remove the last item
print(thislist) # ["apple", "banana"]

thislist = ["apple", "banana", "cherry"]
del thislist[0] # delete the first item
print(thislist) # ['banana', 'cherry']

thislist = ["apple", "banana", "cherry"]
del thislist

thislist = ["apple", "banana", "cherry"]
thislist.clear()
print(thislist) # []