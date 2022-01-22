import random
import array as arr

"""
Array vs List in Python
-----------------------
An array is a vector containing homogeneous elements i.e. belonging to the same data type. 
Elements are allocated with contiguous memory locations allowing easy modification, that is, addition, deletion, accessing of elements. 
In Python, we have to use the array module to declare arrays. 
If the elements of an array belong to different data types, an exception “Incompatible data types” is thrown.

In all other programming languages, you define the size of an array when you instantiate it. That is not true in Python.
In Python, the only difference between array and list is that array is strictly typed, whereas list is not. 

A list in Python is a collection of items which can contain elements of multiple data types, which may be either numeric, character logical values, etc. 
It is an ordered collection supporting negative indexing. A list can be created using [] containing data values.
Contents of lists can be easily merged and copied using python’s inbuilt functions.

https://www.geeksforgeeks.org/difference-between-list-and-array-in-python/#:~:text=Output%3A%201%202%203%20Here%20are%20the%20differences,elements%20%20...%20%204%20more%20rows%20
"""

randomnumber = random.randint(2, 7)

# array with int type
a = arr.array('i', [])

# listofnumbers = []  # you can create a list object in this way or by using a constructor list()
listofnumbers = list()
for index in range(0, randomnumber):
    n = int(input("Enter number: "))
    listofnumbers.append(n)
    a.append(n)


# using list
biggest = listofnumbers[0]
for index in range(1, len(listofnumbers)):
    nextnumber = listofnumbers[index]
    if nextnumber > biggest:
        biggest = nextnumber

""" Using array 
biggest = a[0]
for index in range(1, len(a)):
    nextnumber = a[index]
    if nextnumber > biggest:
        biggest = nextnumber
"""

print("Biggest number is ", biggest)