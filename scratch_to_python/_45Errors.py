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

"""
The try block lets you test a block of code for errors.

The except block lets you handle the error.

The else block lets you execute code when there is no error.

The finally block lets you execute code, regardless of the result of the try- and except blocks.
"""
"""
# Example
try:
  print(x)
except:
  print("An exception occurred")
"""
"""
Many Exceptions

try:
    print(x)
except NameError:
    print("Variable x is not defined")
except:
    print("Something else went wrong")
"""

"""
You can use the else keyword to define a block of code to be executed if no errors were raised

try:
  print("Hello")
except:
  print("Something went wrong")
else:
  print("Nothing went wrong")
"""

"""
The finally block, if specified, will be executed regardless if the try block raises an error or not.

try:
  print(x)
except:
  print("Something went wrong")
finally:
  print("The 'try except' is finished")
"""

"""
Nested try-except blocks

try:
  f = open("demofile.txt")
  try:
    f.write("Lorum Ipsum")
  except:
    print("Something went wrong when writing to the file")
  finally:
    f.close()
except:
  print("Something went wrong when opening the file")
"""

"""
Raise an exception

x = -1

if x < 0:
    raise Exception("Sorry, no numbers below zero")


x = "hello"

if not type(x) is int:
    raise TypeError("Only integers are allowed")
"""

# https://docs.python.org/3/tutorial/errors.html#tut-userexceptions

"""
try:
     raise Exception('spam', 'eggs')
 except Exception as inst:  # creating an alias
     print(type(inst))    # the exception instance
     print(inst.args)     # arguments stored in .args
     print(inst)          # __str__ allows args to be printed directly,
                          # but may be overridden in exception subclasses
     x, y = inst.args     # unpack args
     print('x =', x)
     print('y =', y)

<class 'Exception'>
('spam', 'eggs')
('spam', 'eggs')
x = spam
y = eggs



def this_fails():
    x = 1/0
    try:
        this_fails()
    except ZeroDivisionError as err:
        print('Handling run-time error:', err)
"""

"""
If you need to determine whether an exception was raised but don’t intend to handle it, a simpler form of the raise statement allows you to re-raise the exception:

try:
    raise NameError('HiThere')
    except NameError:
        print('An exception flew by!')
        raise

An exception flew by!
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
NameError: HiThere
"""

"""
Exception Chaining:
The raise statement allows an optional from which enables chaining exceptions. For example:

# exc must be exception instance or None.
raise RuntimeError from exc
This can be useful when you are transforming exceptions. For example:

def func():
    raise ConnectionError

try:
    func()
except ConnectionError as exc:
    raise RuntimeError('Failed to open database') from exc

ConnectionError

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
RuntimeError: Failed to open database
"""
"""
Exception chaining happens automatically when an exception is raised inside an except or finally section. 
This can be disabled by using from None idiom.

try:
    open('database.sqlite')
except OSError:
    raise RuntimeError from None

Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
RuntimeError
"""
