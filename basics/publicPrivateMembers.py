# https://www.geeksforgeeks.org/underscore-_-python/
# https://www.geeksforgeeks.org/role-of-underscores-_-in-python/?ref=next_article
# variable name without prefix underscore is public
# variable name with one prefix underscore is protected
# variable name with two prefix underscore is private

# variable name with one postfix underscore is for reserved word

# __ before the variable name is used for Mangling
# https://medium.com/pythoniq/what-is-name-mangling-in-python-e40403b4048e


'''
Dunder methods
--------------
https://realpython.com/python-double-underscore/#dunder-names-in-python

In Python, names with double leading and trailing underscores (__) have special meaning to the language itself. These names are known as dunder names, and dunder is short for double underscore. In most cases, Python uses dunder names for methods that support a given functionality in the language’s data model.

Dunder methods are also known as special methods, and in some informal circles, they’re called magic methods. Why magic? Because Python calls them automatically in response to specific actions. For example, when you call the built-in len() function with a list object as an argument, Python calls list.__len__() under the hood to retrieve the list’s length.

In general, dunder names are reserved for supporting internal Python behaviors. So, you should avoid inventing such names. Instead, you should only use documented dunder names. In the end, creating a custom dunder name won’t have a practical effect because Python only calls those special methods that the language defines.

'''

class Gfg:
    a = None
    _b = None
    __c = None

    # Constructor
    def __init__(self, a, b, c):
        # Data members
        # Public
        self.a = a

        # Protected
        self._b = b

        # Private
        self.__c = c

    # Methods
    # Private method
    def __display(self):
        print(self.a)
        print(self._b)
        print(self.__c)

    def __display__(self):
        print(self.a)
        print(self._b)
        print(self.__c)

    # Public method
    def accessPrivateMethod(self):
        self.__display()


# Driver code
# Creating object
Obj = Gfg('Geeks', 4, "Geeks!")

# Calling method
Obj.accessPrivateMethod()
Obj.__display__()
