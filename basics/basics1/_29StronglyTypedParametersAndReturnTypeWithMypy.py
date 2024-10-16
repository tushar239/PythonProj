"""
Function Annotations
https://www.geeksforgeeks.org/function-annotations-python/

Function annotations are arbitrary python expressions that are associated with various part of functions.
These expressions are evaluated at compile time and have no life in python’s runtime environment. Python does not attach any meaning to these annotations.
They take life when interpreted by third party libraries, for example, 'mypy'.

Syntax of function annotations
    def foobar(a: expression, b: expression = 5):

    The word ‘expression’ mentioned below can be the type of the parameters that should be passed or comment or any arbitrary string that can be made use by external libraries in a meaningful way.

https://mypy.readthedocs.io/en/stable/introduction.html
What is Mypy?
    Mypy is a static type checker for Python 3 and Python 2.7. If you sprinkle your code with type annotations, mypy can type check your code and find common bugs.
    As mypy is a static analyzer, or a lint-like tool, the type annotations are just hints for mypy and don’t interfere when running your program.
    You run your program with a standard Python interpreter, and the annotations are treated effectively as comments.

Just like Java generics, this type checking has no life during runtime, it is checked at compile/interpretation time only.
"""
from typing import Optional

"""
Go to Settings -> External Tool and create a tool with any name
    Program: C:\Projects\PythonProject\venv\Scripts\mypy.exe
    Argument: $ProjectFileDir$\scratch_to_python\    (this will run mypy for all the files in this directory)
            OR  $ProjectFileDir$\scratch_to_python\<filename>.py    (this will run mypy for a specified file only)
    Working Directory: $ProjectFileDir$
Right click this file and run this external tool to interpret your program using mypy.
And then run your program.

OR

in Terminal, 
    mypy <filepath>.py  --- This will interpret your program using mypy
Run your program 
"""


# def foobar(a: 'int') -> 'int':
# OR
def foobar(a: int) -> int:
    return a


# # If you interprete using normal python interpreter, then following line will just work fine, even if you are expecting a strong type checking of parameters and return type.
# print(foobar("str"))  # error: Argument 1 to "foobar" has incompatible type "str"; expected "int"

print(foobar(9))

# age:int = 'a'  # mypy will throw an error

name = "Steve"  # this will automatically assign str type to this variable
# name = 12 # mypy will throw an error - error: Incompatible types in assignment (expression has type "int", variable has type "str")

# https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html

# strong typing for different data types
x: list[int] = [1]
x: set[int] = {6, 7}

x: dict[str, float] = {"field": 2.0}

x: tuple[int, str, float] = (3, "yes", 7.5)
# For tuples of variable size, we use one type and ellipsis
x: tuple[int, ...] = (1, 2, 3)


def some_function():
    pass

# Use Optional[] for values that could be None
x: Optional[str] = some_function()
# Mypy understands a value can't be None in an if-statement
if x is not None:
    print(x.upper())
# If a value can never be None due to some invariants, use an assert
assert x is not None
print(x.upper())
