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

############## Looping the list ###############
# Looping by item (like a smart for loop in java)
thislist = ["apple", "banana", "cherry"]
for x in thislist:
  print(x)

# Looping by index number (like an old for loop in java)
thislist = ["apple", "banana", "cherry"]
for i in range(len(thislist)):
  print(thislist[i])

thislist = ["apple", "banana", "cherry"]
i = 0
while i < len(thislist):
  print(thislist[i])
  i = i + 1

################ List comprehension ################

'''
syntax
newlist = [expression for item in list if condition == True]
'''

thislist = ["apple", "banana", "cherry"]
[print(x) for x in thislist]

newlist = [x for x in thislist]
print(newlist) # ['apple', 'banana', 'cherry']


fruits = ["apple", "banana", "cherry", "kiwi", "mango"]
newlist = []
for x in fruits:
  if "a" in x:
    newlist.append(x)
print(newlist)
# is same as
newlist = [x for x in fruits if "a" in x]
print(newlist)

fruits = ["apple", "banana", "cherry", "kiwi", "mango"]
newlist = []
for x in fruits:
    if x != 'banana':
        newlist.append(x)
    else:
        newlist.append('orange')
print(newlist) # ['apple', 'orange', 'cherry', 'kiwi', 'mango']
# is same as
newlist = [x if x != "banana" else "orange" for x in fruits]
print(newlist) # ['apple', 'orange', 'cherry', 'kiwi', 'mango']

newlist = [x for x in fruits if x != "apple"]

newlist = [x for x in range(10)]

newlist = [x for x in range(10) if x < 5]

newlist = [x.upper() for x in fruits]

newlist = ['hello' for x in fruits]

##################### Sort the list #######################
thislist = ["orange", "mango", "kiwi", "pineapple", "banana"]
thislist.sort()
print(thislist)

thislist = [100, 50, 65, 82, 23]
thislist.sort()
print(thislist)

# sort descending
thislist = ["orange", "mango", "kiwi", "pineapple", "banana"]
thislist.sort(reverse = True)
print(thislist)

thislist = [100, 50, 65, 82, 23]
thislist.sort(reverse = True)
print(thislist)

#################### copy the list ####################
thislist = ["apple", "banana", "cherry"]
mylist = thislist.copy()
print(mylist)

thislist = ["apple", "banana", "cherry"]
mylist = list(thislist)
print(mylist)

thislist = ["apple", "banana", "cherry"]
mylist = thislist[:]
print(mylist)

###################### Join two lists ###################
list1 = ["a", "b", "c"]
list2 = [1, 2, 3]

list3 = list1 + list2
print(list3)

list1 = ["a", "b" , "c"]
list2 = [1, 2, 3]

for x in list2:
  list1.append(x)
print(list1)

list1 = ["a", "b" , "c"]
list2 = [1, 2, 3]

list1.extend(list2)
print(list1)