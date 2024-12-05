"""
List items are ordered, changeable, and allow duplicate values.

List items are indexed, the first item has index [0], the second item has index [1] etc.
Remember that the first item has index 0.

The list is changeable, meaning that we can change, add, and remove items in a list after it has been created.

https://techvidvan.com/tutorials/python-lists/

Lists are ordered. We can index them and access values. We will see how to do this in the next heading.
Lists are heterogeneous. A list can contain different types of elements.
Lists are mutable. You can change values in them.

Python List vs Numpy Array
https://www.geeksforgeeks.org/python-lists-vs-numpy-arrays/

Python List is like a LinkedList.
Numpy array ls like an ArrayList

Element Overhead: Lists in Python store additional information about each element, such as its type and reference count. This overhead can be significant when dealing with a large number of elements.
Datatype: Lists can hold different data types, but this can decrease memory efficiency and slow numerical operations.
Memory Fragmentation: Lists may not store elements in contiguous memory locations, causing memory fragmentation and inefficiency.
Performance: Lists are not optimized for numerical computations and may have slower mathematical operations due to Python’s interpretation overhead. They are generally used as general-purpose data structures.
Functionality: Lists can store any data type, but lack specialized NumPy functions for numerical operations.

Python List vs Numpy Array
Python list is like LinkedList whereas Numpy Array is like ArrayList
Python list can contain different type of data whereas Numpy array can contain same type of data only.
Numpy array has faster mathematical operations compared to Python List.
"""

thislist = ["apple", "banana", "cherry", "apple", "cherry"]
print(thislist)  # ['apple', 'banana', 'cherry', 'apple', 'cherry']

# Accessing the list
print(thislist[1])  # banana
print(thislist[-1])  # cherry

# Range of indexes
thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
print(thislist[2:5])  # ['cherry', 'orange', 'kiwi']

thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
print(thislist[:4])  # ['apple', 'banana', 'cherry', 'orange']
print(thislist[2:])  # ['cherry', 'orange', 'kiwi', 'melon', 'mango']
print(thislist[-4:-1])  # ['orange', 'kiwi', 'melon']

# Check if item Exists
thislist = ["apple", "banana", "cherry"]
if "apple" in thislist:     # same as thislist.contains("apple") in Java
    print("Yes, 'apple' is in the fruits list")

# Change Item Value
thislist = ["apple", "banana", "cherry"]
thislist[1] = "blackcurrant"
print(thislist)  # ['apple', 'blackcurrant', 'cherry']

# Change a Range of Item Values
thislist = ["apple", "banana", "cherry", "orange", "kiwi", "mango"]
thislist[1:3] = ["blackcurrant", "watermelon"]
print(thislist)  # ['apple', 'blackcurrant', 'watermelon', 'orange', 'kiwi', 'mango']

# If you insert more items than you replace, the new items will be inserted where you specified, and the remaining items will move accordingly:
thislist = ["apple", "banana", "cherry"]
thislist[1:2] = ["blackcurrant", "watermelon"]
print(thislist)  # ['apple', 'blackcurrant', 'watermelon', 'cherry']


# If you insert less items than you replace, the new items will be inserted where you specified, and the remaining items will move accordingly:
thislist = ["apple", "banana", "cherry"]
thislist[1:3] = ["watermelon"]
print(thislist)  # ['apple', 'watermelon']

# Insert Items
thislist = ["apple", "banana", "cherry"]
thislist.insert(2, "watermelon")
print(thislist)  # ['apple', 'banana', 'watermelon', 'cherry']

# Inserting Items at the end
thislist = ["apple", "banana", "cherry"]
thislist.append("orange")
#thislist.insert(len(thislist), "orange")
print(thislist)  # ['apple', 'banana', 'cherry', 'orange']

# Looping through list
for element in thislist:
    print(element, end=", ")  # apple, banana, cherry, orange,

print()

# This is how you can access the list in reverse order
# Unlike to Set, you can access the List using index because it preserves the insertion order
for index in range(len(thislist) - 1, -1, -1):
    print(thislist[index], end=", ")  # orange, cherry, banana, apple,

print()

# Extend list - To append elements from another list to the current list, use the extend() method
thislist = ["apple", "banana", "cherry"]
tropical = ["mango", "pineapple", "papaya"]
thislist.extend(tropical)
print(thislist)  # ['apple', 'banana', 'cherry', 'mango', 'pineapple', 'papaya']

# Add any iterable
thislist = ["apple", "banana", "cherry"]
thistuple = ("kiwi", "orange")
thislist.extend(thistuple)
print(thislist)  # ['apple', 'banana', 'cherry', 'kiwi', 'orange']

# Remove 'banana'
thislist = ["apple", "banana", "cherry"]
thislist.remove("banana")
print(thislist)  # ['apple', 'cherry']

# Remove specified indexed element
# You can't pop an element by index in Set because access order is not same as insertion order in the Set.
thislist = ["apple", "banana", "cherry"]
thislist.pop(1)
print(thislist)  # ['apple', 'cherry']

# If you do not specify the index, the pop() method removes the last item.
thislist = ["apple", "banana", "cherry"]
thislist.pop()
print(thislist)  # ['apple', 'banana']

# The del keyword also removes the specified index:
thislist = ["apple", "banana", "cherry"]
del thislist[0]  # same as thislist.pop(0)
print(thislist)  # ['banana', 'cherry']

# delete the entire list
thislist = ["apple", "banana", "cherry"]
del thislist
# print(thislist)  # NameError: name 'thislist' is not defined

# Loop through a list
thislist = ["apple", "banana", "cherry"]
for x in thislist:
    print(x, end=", ")

print()  # apple, banana, cherry,

# Loop through the index numbers
thislist = ["apple", "banana", "cherry"]
for i in range(len(thislist)):
    print(thislist[i], end=", ")

print()  # apple, banana, cherry,

# Loop through the index numbers in reverse direction
for i in range(len(thislist)-1, -1, -1):
    print(thislist[i], end=", ")  # cherry, banana, apple,

print()

# See ListComprehension.py
# nested for loops and if loop
newlist = [x.upper() for i in range(len(thislist)) for x in thislist[i] if 'a' in thislist[i]]
print(newlist)  # ['A', 'P', 'P', 'L', 'E', 'B', 'A', 'N', 'A', 'N', 'A']
