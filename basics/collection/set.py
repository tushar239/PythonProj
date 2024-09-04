# A set is a collection which is unordered, UNCHANGEABLE*, and unindexed.
# Set items are UNCHANGEABLE, but you can remove items and add new items.

thisset = {"apple", "banana", "cherry", True, 1, 2}
print(thisset) # {'apple', 2, True, 'cherry', 'banana'}
print(len(thisset)) # 5

# creating a set using set() constructor
thisset = set(("apple", "banana", "cherry")) # note the double round-brackets

########################### Accessing items from a set ########################
thisset = {"apple", "banana", "cherry"}
for x in thisset:
  print(x)

thisset = {"apple", "banana", "cherry"}
print("banana" in thisset) # True
print("banana" not in thisset) # False

################### changing the item in a set ################

# can't be changed
# thisset[1] = 'new banana' # set' object does not support item assignment

# insert is also not allowed
#thisset.insert(1, "kiwi")
#print(thisset)

# add, remove, update(same as extend) is allowed
thisset = {"apple", "banana", "cherry"}
thisset.add("orange")
print(thisset) # {'apple', 'orange', 'banana', 'cherry'}

thisset = {"apple", "banana", "cherry"}
tropical = {"pineapple", "mango", "papaya"}
thisset.update(tropical) # set doesn't have extend(). It has update().
print(thisset) # {'apple', 'papaya', 'pineapple', 'mango', 'cherry', 'banana'}

'''
# + operator not allowed to join two sets
thisset = {"apple", "banana", "cherry"}
tropical = {"pineapple", "mango", "papaya"}
thisset = thisset + tropical
print(thisset)
'''

thisset = {"apple", "banana", "cherry"}
mylist = ["kiwi", "orange"]
thisset.update(mylist) # {'banana', 'cherry', 'apple', 'kiwi', 'orange'}
print(thisset)

# remove() or discard() can be used to remove an item
# If the item to remove does not exist, remove() will raise an error.
# If the item to remove does not exist, discard() will NOT raise an error.
thisset = {"apple", "banana", "cherry"}
thisset.remove("banana")
print(thisset) # {'cherry', 'apple'}

thisset = {"apple", "banana", "cherry"}
thisset.discard("banana")
print(thisset) # {'cherry', 'apple'}

# Remove a RANDOM item by using the pop() method.
thisset = {"apple", "banana", "cherry"}
x = thisset.pop()
print(x)
print(thisset)

thisset = {"apple", "banana", "cherry"}
thisset.clear()
print(thisset) # set()

thisset = {"apple", "banana", "cherry"}
del thisset
# print(thisset)

##################### Loop the set #######################
thisset = {"apple", "banana", "cherry"}
for x in thisset:
  print(x)
# comprehension method. Read list.py
[print(x) for x in thisset]

################# Join sets #########################
#The union() and update() methods joins all items from both sets.
#The intersection() method keeps ONLY the duplicates.
#The difference() method keeps the items from the first set that are not in the other set(s).
#The symmetric_difference() method keeps all items EXCEPT the duplicates.

set1 = {"a", "b", "c"}
set2 = {1, 2, 3}
set3 = set1.union(set2)
print(set3) # {1, 2, 3, 'b', 'c', 'a'}

# you can use | operator also instead of union()/update()
set1 = {"a", "b", "c"}
set2 = {1, 2, 3}
set3 = set1 | set2
print(set3) # {1, 2, 3, 'b', 'c', 'a'}

# join multiple sets
set1 = {"a", "b", "c"}
set2 = {1, 2, 3}
set3 = {"John", "Elena"}
set4 = {"apple", "bananas", "cherry"}
myset = set1.union(set2, set3, set4)
print(myset) # {'John', 1, 2, 'b', 'c', 3, 'a', 'apple', 'bananas', 'cherry', 'Elena'}

set1 = {"a", "b", "c"}
set2 = {1, 2, 3}
set3 = {"John", "Elena"}
set4 = {"apple", "bananas", "cherry"}
myset = set1 | set2 | set3 |set4
print(myset)

#  The  | operator only allows you to join sets with sets, and not with other data types like you can with the  union() method.
# join a set and a tuple
x = {"a", "b", "c"}
y = (1, 2, 3)
z = x.union(y)
print(z) # {'c', 2, 1, 3, 'b', 'a'}

set1 = {"a", "b" , "c"}
set2 = {1, 2, 3}
set1.update(set2)
print(set1) # {'c', 'a', 1, 2, 3, 'b'}

set1 = {"apple", "banana", "cherry"}
set2 = {"google", "microsoft", "apple"}
set3 = set1.intersection(set2)
print(set3) # {'apple'}

# you can use & operator also instead of intersection()
set1 = {"apple", "banana", "cherry"}
set2 = {"google", "microsoft", "apple"}
set3 = set1 & set2
print(set3) # {'apple'}

# Keep the items that exist in both set1, and set2
set1 = {"apple", "banana", "cherry"}
set2 = {"google", "microsoft", "apple"}
set1.intersection_update(set2)
print(set1)

# The values True and 1 are considered the same value. The same goes for False and 0
set1 = {"apple", 1,  "banana", 0, "cherry"}
set2 = {False, "google", 1, "apple", 2, True}
set3 = set1.intersection(set2)
print(set3) # {False, 1, 'apple'}

# Keep all items from set1 that are not in set2
set1 = {"apple", "banana", "cherry"}
set2 = {"google", "microsoft", "apple"}
set3 = set1.difference(set2)
print(set3) # {'banana', 'cherry'}

# you can use - operator also instead of difference()
set1 = {"apple", "banana", "cherry"}
set2 = {"google", "microsoft", "apple"}
set3 = set1 - set2
print(set3) # {'banana', 'cherry'}

# The difference_update() method will also keep the items from the first set that are not in the other set,
# but it will change the original set instead of returning a new set.
set1 = {"apple", "banana", "cherry"}
set2 = {"google", "microsoft", "apple"}
set1.difference_update(set2)
print(set1) # {'banana', 'cherry'}

# Keep the items that are not present in both sets:
set1 = {"apple", "banana", "cherry"}
set2 = {"google", "microsoft", "apple"}
set3 = set1.symmetric_difference(set2)
print(set3) # {'banana', 'microsoft', 'google', 'cherry'}

#You can use the ^ operator instead of the symmetric_difference() method
set1 = {"apple", "banana", "cherry"}
set2 = {"google", "microsoft", "apple"}
set3 = set1 ^ set2
print(set3) # {'banana', 'microsoft', 'google', 'cherry'}

# Use the symmetric_difference_update() method to keep the items that are not present in both sets:
set1 = {"apple", "banana", "cherry"}
set2 = {"google", "microsoft", "apple"}
set1.symmetric_difference_update(set2)
print(set1) # {'banana', 'google', 'cherry', 'microsoft'}
