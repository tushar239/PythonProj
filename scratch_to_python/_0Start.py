
"""
Python is written in C language and the name of the library is Cpython.
Source code of Cpython - https://github.com/python/cpython
At present, python's 3.x version is available in CPython.

Later on Python was also written in Java and the name of the library is Jython.
Source code of Jython - https://github.com/jython/jython
At present, python's 2.x version is available in Jython.

By default, we use CPython.

Videos for kids - Variables and Data Types - https://www.youtube.com/watch?v=mQamOwiW3iM
                  String manipulation      - https://www.youtube.com/watch?v=BEMoUK9BBIA
                  Type conversion          - https://www.youtube.com/watch?v=8bVyl_MHRBQ
                  List                     - https://www.youtube.com/watch?v=fAr6EMp0SSc


Interpreter vs Compiler
-----------------------
Python uses interpreter. Java, C, C++ uses compiler
Difference between interpreter and compiler - https://www.geeksforgeeks.org/difference-between-compiler-and-interpreter/

Compiler converts the all source code in machine understandable code, whereas interpreter does that line by line.
Everytime, you run the program, it interprets and coverts the code into machine understandable code line by line.
Interpreter is slower than Compiler.

Difference between == and is
----------------------------
The Equality operator (==) compares the values of both the operands and checks for value equality.
Whereas the 'is' operator checks whether both the operands refer to the same object or not (present in the same memory location)

In Java, obj1.equals(obj2) function compares the value of objects and == operator compares the memory locations of the object.

String is immutable in Python and Java both
-------------------------------------------
Strings are not mutable in Python. Strings are a immutable data types which means that its value cannot be updated.
https://www.tutorialspoint.com/python_text_processing/python_string_immutability.htm

No literals in Python
---------------------
Everything is an object in python. There is no concept of literals like Java in Python.
so, int,float,string everything is an object.

Python Collections
------------------
List is a collection which is ordered and changeable. Allows duplicate members.
Tuple is a collection which is ordered and unchangeable. Allows duplicate members.
Set is a collection which is unordered, unchangeable*, and unindexed. No duplicate members.
Dictionary is a collection which is ordered** and changeable. No duplicate keys.
"""

"""
    Variables
    1. It can have any name
    2. It can have any value of any data type
    3. The value and data type can change anytime
    
    Float in python is same as Double in Java
"""

# String processing
s: str = 'Hello ' + 'I am ' + 'Emma'
print(type(s))  # <class 'str'>
print(s)  # Hello I am Emma
print(len(s))  # 15
print(s[1])  # e
print(s[1:3])  # el         --- substring feature
print(s[-4:])  # Emma       --- you can also evaluate the string from the end. Index from the end starts from -1.

# complete print function parameters usages
print("hi", "hello", sep="@", end="!\n")  # hi@hello!      default end=\n

# type() function
a = 5
print("Data type of a is: ", type(a))  # <class 'int'>
a = 5.2
print("Data type of a is: ", type(a))  # <class 'float'>
a = "January"
print("Data type of a is: ", type(a))  # <class 'str'>

# int(), float(), str() are type conversion functions
a = "100"
# print(a+15)  # TypeError: can only concatenate str (not "int") to str
print(int(a)+15)  # 115

# isdigit() function
print(a.isdigit())  # true  or str.isdigit(a)

b = None  # it is like null in java
# print(b.isdigit())  # AttributeError: 'NoneType' object has no attribute 'isdigit'
# print(str.isdigit(b))  # TypeError: descriptor 'isdigit' for 'str' objects doesn't apply to a 'NoneType' object

# list processing
lunch = ["pizza", "pasta", "hamburger", "pancake", "doughnut"]
print(lunch[2])  # hamburger
for num in range(len(lunch)):
    print(lunch[num])


# To use random number
import random  # this is not a good practice to add an import statement in between the code. Always put on the top. But it works.

print(random.choice(lunch))
random.random()
