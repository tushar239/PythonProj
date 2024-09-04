################  Create a tuple ##################

# A tuple is a collection which is ordered and UNCHANGEABLE.

tuple1 = ("apple", "banana", "cherry")
tuple2 = (1, 5, 7, 9, 3)
tuple3 = (True, False, False)

# duplicates are allowed
thistuple = ("apple", "banana", "cherry", "apple", "cherry")
print(thistuple)

# length of tuple
thistuple = ("apple", "banana", "cherry")
print(len(thistuple))

# IMPORTANT: To create a tuple with only one item, you have to add a comma after the item,
# otherwise Python will not recognize it as a tuple.

thistuple = ("apple",) # <class 'tuple'>
print(type(thistuple))
print(thistuple) # ('apple',)

#NOT a tuple
thistuple = ("apple")
print(type(thistuple)) # <class 'str'>
print(thistuple) # apple

# creating a tuple using tuple() constructor
thistuple = tuple(("apple", "banana", "cherry")) # note the double round-brackets
print(thistuple)

################## Access tuple items ######################

thistuple = ("apple", "banana", "cherry")
print(thistuple[1]) # banana

# negative indexing
thistuple = ("apple", "banana", "cherry")
print(thistuple[-1]) # cherry

thistuple = ("apple", "banana", "cherry", "orange", "kiwi", "melon", "mango")
print(thistuple[2:5])
print(thistuple[:4])
print(thistuple[2:])
print(thistuple[-4:-1])

thistuple = ("apple", "banana", "cherry")
if "apple" in thistuple:
  print("Yes, 'apple' is in the fruits tuple")

############# change tuple values ######################
# tuple is UNCHANGEABLE.
# If you want to change anything in it, you have to first create a list from that tuple, change it and then covert that list back to tuple.
x = ("apple", "banana", "cherry")
y = list(x)
y[1] = "kiwi"
x = tuple(y)
print(x) # ('orange', 'kiwi', 'melon')

thistuple = ("apple", "banana", "cherry")
y = list(thistuple)
y.append("orange")
thistuple = tuple(y)
print(thistuple)  # ('apple', 'kiwi', 'cherry', 'orange')

thistuple = ("apple", "banana", "cherry")
y = ("orange",)
thistuple += y
print(thistuple) # ('apple', 'banana', 'cherry', 'orange')

##################### Remove an item from tuple #####################
thistuple = ("apple", "banana", "cherry")
y = list(thistuple)
y.remove("apple")
thistuple = tuple(y)
print(thistuple)

thistuple = ("apple", "banana", "cherry")
del thistuple
#print(thistuple) #this will raise an error because the tuple no longer exists

####################### Unpacking a tuple ###################
fruits = ("apple", "banana", "cherry")
(green, yellow, red) = fruits
print(green) # apple
print(yellow) # banana
print(red) # cherry

fruits = ("apple", "banana", "cherry", "strawberry", "raspberry")
(green, yellow, *red) = fruits
print(green) # apple
print(yellow) # banana
print(red) # ['cherry', 'strawberry', 'raspberry']

fruits = ("apple", "mango", "papaya", "pineapple", "cherry")
(green, *tropic, red) = fruits
print(green) # apple
print(tropic) # ['mango', 'papaya', 'pineapple']
print(red) # cherry

######################### Loop Tuple #######################
# Looping by item (like a smart for loop in java)
thistuple = ("apple", "banana", "cherry")
for x in thistuple:
  print(x)

# Looping by index number (like an old for loop in java)
thistuple = ("apple", "banana", "cherry")
for i in range(len(thistuple)):
  print(thistuple[i])

thistuple = ("apple", "banana", "cherry")
i = 0
while i < len(thistuple):
  print(thistuple[i])
  i = i + 1

################# Joining two tuples ######################
tuple1 = ("a", "b" , "c")
tuple2 = (1, 2, 3)
tuple3 = tuple1 + tuple2
print(tuple3) # ('a', 'b', 'c', 1, 2, 3)

fruits = ("apple", "banana", "cherry")
mytuple = fruits * 2
print(mytuple) # ('apple', 'banana', 'cherry', 'apple', 'banana', 'cherry')


