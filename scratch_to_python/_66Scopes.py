"""
Scopes:
    Local
    Global
"""

"""
Local Scope
A variable created inside a function belongs to the local scope of that function, and can only be used inside that function.
"""


# A variable created inside a function is available inside that function:
def myfunc():
    x = 300
    print(x)

myfunc()  # 300

# The local variable can be accessed from a function within the function:
def myfunc():
    x = 300

    def myinnerfunc():
        print(x)

    myinnerfunc()


myfunc()  # 300

# A variable created in the main body of the Python code is a global variable and belongs to the global scope.
# Global variables are available from within any scope, global and local.
x = 300

def myfunc():
    print(x)

myfunc()  # 300

print(x)  # 300

# IMP:  If you operate with the same variable name inside and outside of a function, Python will treat them as two separate variables,
#       one available in the global scope (outside the function) and one available in the local scope (inside the function):
# This is different from Java.

x = 300

def myfunc():
    x = 200
    print(x)

myfunc()  # 200

print(x)  # 300

# To change the value of a global variable inside a function, refer to the variable by using the global keyword:

x = 300

def myfunc():
    global x
    x = 200

myfunc()

print(x)  # 200

# If you need to create a global variable, but are stuck in the local scope, you can use the global keyword.
# The global keyword makes the variable global.
def myfunc():
    global x
    x = 300

myfunc()

print(x)  # 300
