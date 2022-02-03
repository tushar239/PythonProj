"""
What is an Exception?
An unwanted or unexpected event which disturbs the normal flow of the program is called exception.
Whenever an exception occurs, then immediately program will terminate abnormally.
In order to get our program executed normally, we need to handle those exceptions on high priority.

https://www.pyforschool.com/tutorial/types-of-errors.html

There are mainly 3 types of errors in Python.
1. Syntax errors

    print "Hello World!"
    In this first example, we forget to use the parenthesis that are required by print(). Python does not understand what you are trying to do.

2. Logical error

    Example: For example, perhaps you want a program to calculate the average of two numbers and get the result like this :

    x = 3
    y = 4
    average = x + y / 2
    print(average)  # 5.0

3. Runtime errors

    Run time errors arise when the python knows what to do with a piece of code but is unable to perform the action.
    Since Python is an interpreted language, these errors will not occur until the flow of control in your program reaches the line with the problem.
    Common example of runtime errors are using an undefined variable or mistyped the variable name.

    day = "Sunday"
    print(Day)
    Output

    Traceback (most recent call last):
      File "C:/Users/91981/AppData/Local/Programs/Python/Python38/hello.py", line 2, in
        print(Day)
    NameError: name 'Day' is not defined

https://docs.python.org/3/library/exceptions.html#bltin-exceptions

In Python, all exceptions must be instances of a class that derives from BaseException. In a try statement with an except clause that mentions a particular class, that clause also handles any exception classes derived from that class (but not exception classes from which it is derived).
Two exception classes that are not related via subclassing are never equivalent, even if they have the same name.
Programmers are encouraged to derive new exceptions from the Exception class or one of its subclasses, and not from BaseException.

All errors in Python are the UNCHECKED type. Exceptions include both checked and unchecked type.
Python is not compiled, so checked exceptions don't make much sense.

Look at 'Exception Hierarchy' image. When an exception don't fall in other built-in exception category, then you should use RuntimeError.
"""

class B(Exception):
    pass


class C(B):
    pass


class D(C):
    pass

# he following code will print B, C, D in that order
for cls in [B, C, D]:
    try:
        raise cls()
    except D:
        print("D")
    except C:
        print("C")
    except B:
        print("B")
# if the except clauses were reversed (with except B first), it would have printed B, B, B — the first matching except clause is triggered.

"""
All exceptions inherit from BaseException, and so it can be used to serve as a wildcard. Use this with extreme caution, since it is easy to mask a real programming error in this way! It can also be used to print an error message and then re-raise the exception (allowing a caller to handle the exception as well):

# import sys --- default

try:
    f = open('myfile.txt')
    s = f.readline()
    i = int(s.strip())
except OSError as err:
    print("OS error: {0}".format(err))
except ValueError:
    print("Could not convert data to an integer.")
except BaseException as err:
    print(f"Unexpected {err=}, {type(err)=}")
    raise
"""

