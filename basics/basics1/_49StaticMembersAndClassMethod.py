"""
https://www.geeksforgeeks.org/g-fact-34-class-or-static-variables-in-python/

This example shows how class variables (or static variables) are shared across all instances.
However, when an instance modifies the class variable, it creates its own copy, which leads to a different behavior.

In Python, a static variable is a variable that is shared among all instances of a class, rather than being unique to each instance. It is also sometimes referred to as a class variable because it belongs to the class itself rather than any particular instance of the class.
Static variables are defined inside the class definition, but outside of any method definitions.
They are typically initialized with a value, just like an instance variable, but they can be accessed and modified through the class itself, rather than through an instance.


Key Differences Between Class Variables in Python and Static Variables in Java/C++
    While class variables in Python and static variables in Java/C++ serve a similar purpose of being shared across all instances, they behave differently when modified through an instance:

    Java/C++ Behavior: When you modify a static variable in Java or C++, the change is reflected across all instances of the class, and they all remain synchronized with the static variable’s value.
    Python Behavior: In Python, if you modify a class variable through an instance, a new instance variable is created. This separates the modified value from the original class variable, which remains unchanged for other instances.

    IMPORTANT: Unlike to Java, in python, class(static) variable can be accessed by Class method or Static method or Instance method.


https://www.geeksforgeeks.org/class-method-vs-static-method-python/

What is Class Method in Python?

    The @classmethod decorator is a built-in function decorator that is an expression that gets evaluated after your function is defined. The result of that evaluation shadows your function definition.
    A class method receives the class as an implicit first argument, just like an instance method receives the instance

    Syntax Python Class Method:

    class C(object):
        @classmethod
        def fun(cls, arg1, arg2, ...):
           ....
    fun: function that needs to be converted into a class method
    returns: a class method for function.

    - A class method is a method that is bound to the class and not the object of the class.
    - They have the access to the state of the class as it takes a class parameter that points to the class and not the object instance.
    - It can modify a class state that would apply across all the instances of the class.
    For example, it can modify a class variable that will be applicable to all the instances.

What is the Static Method in Python?

    A static method does not receive an implicit first argument. A static method is also a method that is bound to the class and not the object of the class. This method can’t access or modify the class state. It is present in a class because it makes sense for the method to be present in class.

    Syntax Python Static Method:

    class C(object):
        @staticmethod
        def fun(arg1, arg2, ...):
            ...
    returns: a static method for function fun.

Class method vs Static Method
    - A class method takes cls as the first parameter while a static method needs no specific parameters.
    - A class method can access or modify the class state while a static method can’t access or modify it.
    - In general, static methods know nothing about the class state. They are utility-type methods that take some parameters and work upon those parameters. On the other hand class methods must have class as a parameter.
    - We use @classmethod decorator in python to create a class method and we use @staticmethod decorator to create a static method in python.

When to use what?
    - We generally use class method to create factory methods.
    Factory methods return class objects ( similar to a constructor ) for different use cases.
    - We generally use static methods to create utility functions.

How to define a class method and a static method?
    To define a class method in python, we use @classmethod decorator, and to define a static method we use @staticmethod decorator.
"""


# Python program to show that the variables with a value
# assigned in class declaration, are class variables

# Class for Computer Science Student
class CSStudent:
    stream = 'cse'  # Class (static) Variable

    def __init__(self, name, roll):
        self.name = name  # Instance Variable
        self.roll = roll  # Instance Variable


# Objects of CSStudent class
a = CSStudent('Geek', 1)
b = CSStudent('Nerd', 2)

# Class variables can be accessed using class name also
print(CSStudent.stream) # prints "cse"

print(a.stream)  # prints "cse"
print(b.stream)  # prints "cse"

print(a.name)  # prints "Geek"
print(b.name)  # prints "Nerd"

print(a.roll)  # prints "1"
print(b.roll)  # prints "2"


# IMPORTANT: Now if we change the stream for just a, it won't be changed for b
a.stream = 'ece'
print(a.stream)  # prints 'ece'
print(b.stream)  # prints 'cse'

# IMPORTANT: To change the stream for all instances of the class we can change it
# directly from the class
CSStudent.stream = 'mech'

print(a.stream)  # prints 'ece'
print(b.stream)  # prints 'mech'

# Python program to demonstrate
# use of class method and static method.
from datetime import date


class Person:
    someclassvariable = 15 # class/static variable
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # a class method to create a Person object by birth year.
    @classmethod
    def fromBirthYear(cls, name, year):
        cls.someclassvariable = 25
        return cls(name, date.today().year - year)

    # a static method to check if a Person is adult or not.
    @staticmethod
    def isAdult(age): # static methods can't access instance variables
        Person.someclassvariable = 35
        return age > 18

    def somemethod(self):
        Person.someclassvariable = 45


person1 = Person('mayank', 21)
person2 = Person.fromBirthYear('mayank', 1996)
print(Person.someclassvariable) # 25

print(person1.age)  # 21
print(person2.age)  # 28
person1.somemethod()
print(Person.someclassvariable) # 45

# print the result
print(Person.isAdult(22))  # True
print(Person.someclassvariable) # 35