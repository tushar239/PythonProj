# Dunder or magic methods in Python

# https://www.geeksforgeeks.org/dunder-magic-methods-python/

'''
Python Magic methods are the methods starting and ending with double underscores ‘__’. They are defined by built-in classes in Python and commonly used for operator overloading.

They are also called Dunder methods, Dunder here means “Double Under (Underscores)”.
'''


'''
Python Magic Methods
Below are the lists of Python magic methods and their uses.

Initialization and Construction

__new__: To get called in an object’s instantiation.
__init__: To get called by the __new__ method.
__del__: It is the destructor.
Numeric magic methods

__trunc__(self): Implements behavior for math.trunc()
__ceil__(self): Implements behavior for math.ceil()
__floor__(self): Implements behavior for math.floor()
__round__(self,n): Implements behavior for the built-in round()
__invert__(self): Implements behavior for inversion using the ~ operator.
__abs__(self): Implements behavior for the built-in abs()
__neg__(self): Implements behavior for negation
__pos__(self): Implements behavior for unary positive 
Arithmetic operators

__add__(self, other): Implements behavior for math.trunc()
__sub__(self, other): Implements behavior for math.ceil()
__mul__(self, other): Implements behavior for math.floor()
__floordiv__(self, other): Implements behavior for the built-in round()
__div__(self, other): Implements behavior for inversion using the ~ operator.
__truediv__(self, other): Implements behavior for the built-in abs()
__mod__(self, other): Implements behavior for negation
__divmod__(self, other): Implements behavior for unary positive 
__pow__: Implements behavior for exponents using the ** operator.
__lshift__(self, other): Implements left bitwise shift using the << operator.
__rshift__(self, other): Implements right bitwise shift using the >> operator.
__and__(self, other): Implements bitwise and using the & operator.
__or__(self, other): Implements bitwise or using the | operator.
__xor__(self, other): Implements bitwise xor using the ^ operator.
String Magic Methods

__str__(self): Defines behavior for when str() is called on an instance of your class.
__repr__(self): To get called by built-int repr() method to return a machine readable representation of a type.
__unicode__(self): This method to return an unicode string of a type.
__format__(self, formatstr): return a new style of string.
__hash__(self): It has to return an integer, and its result is used for quick key comparison in dictionaries.
__nonzero__(self): Defines behavior for when bool() is called on an instance of your class. 
__dir__(self): This method to return a list of attributes of a class.
__sizeof__(self): It return the size of the object.
Comparison magic methods

__eq__(self, other): Defines behavior for the equality operator, ==.
__ne__(self, other): Defines behavior for the inequality operator, !=.
__lt__(self, other): Defines behavior for the less-than operator, <.
__gt__(self, other): Defines behavior for the greater-than operator, >.
__le__(self, other): Defines behavior for the less-than-or-equal-to operator, <=.
__ge__(self, other): Defines behavior for the greater-than-or-equal-to operator, >=.

'''