"""
https://www.geeksforgeeks.org/destructors-in-python/?ref=lbp

Destructors are called when an object gets destroyed.
In Python, destructors are not needed as much as in C++ because Python has a garbage collector that handles memory management automatically.
The __del__() method is a known as a destructor method in Python. It is called when all references to the object have been deleted i.e when an object is garbage collected.
"""


"""
The destructor is called after the program ended or when all the references to object are deleted i.e when the reference count becomes zero, not when object went out of scope.
"""
class Employee:

    # Initializing (Constructor)
    def __init__(self):
        print('Employee created')

    # Calling destructor
    def __del__(self):
        print("Destructor called")


def Create_obj():
    print('Making Object...')
    obj = Employee()
    print('function end...')
    return obj


print('Calling Create_obj() function...')
obj = Create_obj()
print('Program End...')

"""
O/P:
Calling Create_obj() function...
Making Object...
Employee created
function end...
Program End...
Destructor called
"""

"""
Here is the simple example of destructor. 
By using del keyword we deleted the all references of object ‘obj’, therefore destructor invoked automatically.
"""


# Python program to illustrate destructor
class Employee:

    # Initializing (Constructor)
    def __init__(self):
        print('Employee created.')

    # Deleting (Calling destructor)
    def __del__(self):
        print('Destructor called, Employee deleted.')


obj = Employee()
del obj
"""
O/P:
Employee created.
Destructor called, Employee deleted.
"""