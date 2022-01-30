"""
List items are ordered, changeable, and allow duplicate values.

List items are indexed, the first item has index [0], the second item has index [1] etc.
Remember that the first item has index 0.

The list is changeable, meaning that we can change, add, and remove items in a list after it has been created.
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


