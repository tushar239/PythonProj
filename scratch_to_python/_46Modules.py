"""
https://www.w3schools.com/python/python_modules.asp
https://www.geeksforgeeks.org/python-modules/?ref=lbp

What is a Module?
Consider a module to be the same as a code library.
The module is a simple Python file that contains collections of functions and global variables and with having a .py extension file.
It is an executable file and to organize all the modules we have the concept called Package in Python.

Module is a file containing a set of functions you want to include in your application.

To create a module just save the code you want in a file with the file extension .py

When the interpreter encounters an import statement, it imports the module if the module is present in the search path.
A search path is a list of directories that the interpreter searches for importing a module.

Package: The package is a simple directory having collections of modules. This directory contains Python modules and
also having __init__.py file by which the interpreter interprets it as a Package instead of a normal directory.
The package is simply a namespace. The package also contains sub-packages inside it.

    Example:

    Student(Package)
    | __init__.py (Constructor)
    | details.py (Module)
    | marks.py (Module)
    | collegeDetails.py (Module)
"""

# You can import entire module or just a class or function inside that module
import _46mymodule as mm
# from _46mymodule import greeting, anotherfunction, someclass
# from _46mymodule import *   --- importing all the names

mm.greeting("Tus")
age = mm.person1["age"]
print(age)
