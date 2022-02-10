"""
https://www.geeksforgeeks.org/g-fact-34-class-or-static-variables-in-python/
https://www.geeksforgeeks.org/class-method-vs-static-method-python/

All objects share class or static variables.
An instance or non-static variables are different for different objects (every object has a copy).

In C++ and Java, we can use static keywords to make a variable a class variable. The variables which don’t have a preceding static keyword are instance variables.
The Python approach is simple; it doesn’t require a static keyword.

All variables which are assigned a value in the class declaration are class variables.
And variables that are assigned values inside methods are instance variables.

Class method vs Static Method
    - A class method takes cls as the first parameter while a static method needs no specific parameters.
    - A class method can access or modify the class state while a static method can’t access or modify it.
    - In general, static methods know nothing about the class state. They are utility-type methods that take some parameters and work upon those parameters. On the other hand class methods must have class as a parameter.
    - We use @classmethod decorator in python to create a class method and we use @staticmethod decorator to create a static method in python.

When to use what?
    - We generally use class method to create factory methods. Factory methods return class objects ( similar to a constructor ) for different use cases.
    - We generally use static methods to create utility functions.

How to define a class method and a static method?
    To define a class method in python, we use @classmethod decorator, and to define a static method we use @staticmethod decorator.
"""


# Python program to show that the variables with a value
# assigned in class declaration, are class variables

# Class for Computer Science Student
class CSStudent:
    stream = 'cse'  # Class Variable

    def __init__(self, name, roll):
        self.name = name  # Instance Variable
        self.roll = roll  # Instance Variable


# Objects of CSStudent class
a = CSStudent('Geek', 1)
b = CSStudent('Nerd', 2)

print(a.stream)  # prints "cse"
print(b.stream)  # prints "cse"
print(a.name)  # prints "Geek"
print(b.name)  # prints "Nerd"
print(a.roll)  # prints "1"
print(b.roll)  # prints "2"

# Class variables can be accessed using class
# name also
print(CSStudent.stream)  # prints "cse"

# Now if we change the stream for just a it won't be changed for b
a.stream = 'ece'
print(a.stream)  # prints 'ece'
print(b.stream)  # prints 'cse'

# To change the stream for all instances of the class we can change it
# directly from the class
CSStudent.stream = 'mech'

print(a.stream)  # prints 'ece'
print(b.stream)  # prints 'mech'

# Python program to demonstrate
# use of class method and static method.
from datetime import date


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # a class method to create a Person object by birth year.
    @classmethod
    def fromBirthYear(cls, name, year):
        return cls(name, date.today().year - year)

    # a static method to check if a Person is adult or not.
    @staticmethod
    def isAdult(age):
        return age > 18


person1 = Person('mayank', 21)
person2 = Person.fromBirthYear('mayank', 1996)

print(person1.age)  # 21
print(person2.age)  # 21

# print the result
print(Person.isAdult(22))  # True
