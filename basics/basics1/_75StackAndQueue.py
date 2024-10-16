"""
https://www.geeksforgeeks.org/stack-class-in-java/

Stack:
Java has a class 'Stack'. Python doesn't have it.
Java has a class 'LinkedList'. Python doesn't have it.

Both Java and Python have Queue like datastructure.

As you know, both Stack and Queue are the implementations of LinkedList internally.
As Python doesn't have LinkedList, we need to use List or Queue to implement Stack.

In Python, class 'collections.deque' is more powerful for fast appends and pops from both the front and back end.
So, you can use this class for both stack and queue.
"""

# ------------- Making use of a list as Stack and Queue ---------------------

aList = ["John", "Steve", "Harry"]
print(aList)  # ['John', 'Steve', 'Harry']

# In both, stack and queue, you add an element at the end
aList.append("Kevin")  # ["John", "Steve", "Harry", "Kevin"]
print("After adding a new element:", aList)

# Now, if you want to make a list act like a Stack, you pop an element from the end (Last-In First-Out)
poppedElement = aList.pop()
print("Popped Element:", poppedElement)  # Kevin
print("After popping:", aList)  # ['John', 'Steve', 'Harry']

# Now, if you want to make a list act like a Queue, you pop an element from the front (First-In First-Out)
"""
IMP:
But When it comes to queue, the below list implementation is not efficient. In queue when pop() is made from the beginning of the list which is slow. This occurs due to the properties of list, which is fast at the end operations but slow at the beginning operations, as all other elements have to be shifted one by one.
So, we prefer the use of queue over list, which was specially designed to have fast appends and pops from both the front and back end.
"""
poppedElement = aList.pop(0)
print("Popped Element: ", poppedElement)  # John
print("After popping:", aList)  # ['Steve', 'Harry']

# ------------- Making use of Queue Data Structure ---------------------
"""
This module implements specialized container datatypes providing alternatives to 
Python's general purpose built-in containers, dict, list, set, and tuple.
"""
from collections import deque

queue = deque(["John", "Steve", "Harry"])
print("Initial Queue:", queue)  # deque(['John', 'Steve', 'Harry'])

queue.append("Kevin")
print("After adding a new element to queue:", queue)  # ["John", "Steve", "Harry", "Kevin"]

poppedElement = queue.popleft()
print("Popped Element:", poppedElement)  # John
print("Queue after popping an element:", queue)  # ["Steve", "Harry", "Kevin"]