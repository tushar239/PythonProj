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
# In List, you can use pop(index), but you can't do that for set because access order is not same as insertion order in set. So, you can't access set by an index.
thisset = {"apple", "banana", "cherry"}
x = thisset.pop()
print(x)  # cherry
print(thisset)  # {'banana', 'apple'}

# delete the set completely
# You can't delete/pop an element by index in Set because access order is not same as insertion order in the Set.
thisset = {"apple", "banana", "cherry"}
del (thisset)
# print(thisset)  # NameError: name 'thisset' is not defined

# Loop (iterate) through the set
thisset = {"apple", "banana", "cherry"}

for x in thisset:
    print(x, end=",")  # cherry,banana,apple,

print("hi")

# Unlike to List, you can't access set by index because set doesn't preserve insertion order
"""
for index in range(len(thisset)-1, -1, -1):
    print(thisset[index])
"""

# Keep the items that exist in both set x and set y
x = {"apple", "banana", "cherry"}
y = {"google", "microsoft", "apple"}

x.intersection_update(y)
print(x)  # {'apple'}

# The intersection() method will return a new set, that only contains the items that are present in both sets
x = {"apple", "banana", "cherry"}
y = {"google", "microsoft", "apple"}

z = x.intersection(y)

print(z)  # {'apple'}

# isdisjoint() returns True if no items in set x is present in set y
x = {"apple", "banana", "cherry"}
y = {"google", "microsoft", "facebook"}

z = x.isdisjoint(y)

print(z)  # True

# The symmetric_difference_update() method will keep only the elements that are NOT present in both sets.
x = {"apple", "banana", "cherry"}
y = {"google", "microsoft", "apple"}

x.symmetric_difference_update(y)

print(x)  # {'google', 'microsoft', 'cherry', 'banana'}

# The symmetric_difference() method will return a new set, that contains only the elements that are NOT present in both sets.
x = {"apple", "banana", "cherry"}
y = {"google", "microsoft", "apple"}

z = x.symmetric_difference(y)

print(z)  # {'microsoft', 'google', 'banana', 'cherry'}

# removes all elements from the set
thisset = {"apple", "banana", "cherry"}
thisset.clear()
print(thisset)  # set() --- empty set

# issubset(), issuperset()
x = {"apple", "banana", "cherry"}
y = {"apple"}
print(y.issubset(x))  # True
print(x.issuperset(y))  # True
