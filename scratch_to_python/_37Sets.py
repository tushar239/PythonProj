"""
Set items are unchangeable, meaning that we cannot change the items after the set has been created.
"""

thisset = {"apple", "banana", "cherry", "apple"}

print(type(thisset))  # <class 'set'>
print(thisset)  # {'apple', 'banana', 'cherry'} --- duplicate is removed
print(len(thisset))  # 3

# It is also possible to use the set() constructor to make a set. You can pass any iterable (list, set, tuple etc) to set() constructor

thisset = set(("apple", "banana", "cherry"))  # note the double round-brackets
print(thisset)  # {'apple', 'cherry', 'banana'}

thisset = set(["apple", "banana", "cherry"])  # note the double round-brackets
print(thisset)  # {'apple', 'cherry', 'banana'}

# A set with strings, integers and boolean values:
set1 = {"abc", 34, True, 40, "male"}
print(set1)  # {True, 34, 'male', 40, 'abc'}

# check if "banana" is present in the set
thisset = {"apple", "banana", "cherry"}
print("banana" in thisset)  # True

# Add an item to set
thisset = {"apple", "banana", "cherry"}
thisset.add("orange")
print(thisset)  # {'apple', 'cherry', 'banana', 'orange'}

# Add one set to another
# The update() method inserts the items in set2 into set1
thisset = {"apple", "banana", "cherry"}
tropical = {"pineapple", "mango", "papaya"}
# thisset.add(tropical)  # TypeError: unhashable type: 'set'
thisset.update(tropical)
print(thisset)  # {'papaya', 'cherry', 'banana', 'mango', 'apple', 'pineapple'}

# Add a list to set
thisset = {"apple", "banana", "cherry"}
mylist = ["kiwi", "orange", "kiwi"]
thisset.update(mylist)
print(thisset)  # {'cherry', 'banana', 'kiwi', 'orange', 'apple'}

# The union() method returns a new set with all items from both sets.
# The update() method inserts the items in set2 into set1
set1 = {"a", "b", "c"}
set2 = {1, 2, 3}

set3 = set1.union(set2)
print(set3)  # {'a', 1, 'b', 3, 2, 'c'}
print(set1 is set3)  # False

# Remove an item from a set, use the remove(), or the discard() method
thisset = {"apple", "banana", "cherry"}
thisset.remove("banana")
print(thisset)  # {'apple', 'cherry'}
thisset.discard("cherry")
print(thisset)  # {'apple'}

# Remove the last item by using the pop() method
thisset = {"apple", "banana", "cherry"}
x = thisset.pop()
print(x)
print(thisset)

# delete the set completely
thisset = {"apple", "banana", "cherry"}
del (thisset)
# print(thisset)  # NameError: name 'thisset' is not defined

# Loop (iterate) through the set
thisset = {"apple", "banana", "cherry"}

for x in thisset:
    print(x, end=",")  # cherry,banana,apple,

