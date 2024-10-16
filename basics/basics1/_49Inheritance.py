"""
https://www.w3schools.com/python/python_inheritance.asp
https://www.geeksforgeeks.org/inheritance-in-python/?ref=lbp
https://www.geeksforgeeks.org/types-of-inheritance-python/

Python Inheritance
Inheritance allows us to define a class that inherits all the methods and properties from another class.

Parent class is the class being inherited from, also called base class.

Child class is the class that inherits from another class, also called derived class.

object class is root of all classes

Multiple inheritance:
When a child class inherits from multiple parent classes, it is called multiple inheritance.
Unlike Java and like C++, Python supports multiple inheritance. We specify all parent classes as a comma-separated list in the bracket.

Multilevel inheritance:
When we have a child and grandchild relationship.

There is no interface concept in core Python, but you can use zope.interface module to achieve the interface features.
https://www.geeksforgeeks.org/python-interface-module/
"""


# Create a parent class
class Person:  # it is same as 'class Person(object)'. object is a parent class for all claases.
    def __init__(self, fname, lname):
        self.firstname = fname
        self.lastname = lname

    def printname(self):
        print(self.firstname, self.lastname)


# Use the Person class to create an object, and then execute the printname method:
x = Person("John", "Doe")
x.printname()


# Create a child class
# To create a class that inherits the functionality from another class, send the parent class as a parameter when creating the child class.
# If there is no __init__() in the child class, it will be inherited from the parent class.
# But if there is an __init() in the child class, parent class' __init__() will not be called automatically from child class' __init__() method. You have to call it explicitly.
class Student(Person):
    """
    # IMP: In Java, if there is a non-default constructor in parent class, you will be forced to add similar constructor in child class, but that is not true for python.
    # By default, python will assume that you have following __init__() method
    def __init__(self, fname, lname):
        super().__init__(fname, lname)
    """

    pass  # Note: Use the pass keyword when you do not want to add any other properties or methods to the class.


x = Student("Mike", "Olsen")
x.printname()

print( issubclass(Student, Person))  # True
print( isinstance(x, Person))  # True
print( isinstance(Student(), Person))  # True

class Student(Person):
    def __init__(self, fname, lname, graduationyear=2021):
        Person.__init__(self, fname, lname)  # this doesn't have to the first statement like Java.
        # OR
        # super().__init__(fname, lname)  # By using the super() function, you do not have to use the name of the parent element, it will automatically inherit the methods and properties from its parent.
        self.graduationyear = graduationyear

    # If you add a method in the child class with the same name as a function in the parent class, the inheritance of the parent method will be overridden.
    def printname(self):
        super().printname()
        print("Welcome", self.firstname, self.lastname, "to the class of", self.graduationyear)

x = Student("Mike", "Olsen", 2019)
x.printname()
"""
O/P:
Mike Olsen
Welcome Mike Olsen to the class of 2019
"""


# Multiple inheritance
class Base1(object):
    def __init__(self):
        self.str1 = "Geek1"
        print("Base1")


class Base2(object):
    def __init__(self):
        self.str2 = "Geek2"
        print("Base2")


class Derived(Base1, Base2):
    def __init__(self):
        # Calling constructors of Base1 and Base2 classes
        Base1.__init__(self)
        Base2.__init__(self)
        print("Derived")

    def printStrs(self):
        print(self.str1, self.str2)


ob = Derived()
ob.printStrs()
"""
O/P:
Base1
Base2
Derived
Geek1 Geek2
"""


# Multilevel inheritance:
# Base or Super class. Note object in bracket.
class Base(object):

    # Constructor
    def __init__(self, name):
        self.name = name

    # To get name
    def getName(self):
        return self.name


# Inherited or Sub class (Note Person in bracket)
class Child(Base):

    # Constructor
    def __init__(self, name, age):
        Base.__init__(self, name)
        self.age = age

    # To get name
    def getAge(self):
        return self.age


# Inherited or Sub class (Note Person in bracket)
class GrandChild(Child):

    # Constructor
    def __init__(self, name, age, address):
        Child.__init__(self, name, age)
        self.address = address

    # To get address
    def getAddress(self):
        return self.address


# Driver code
g = GrandChild("Geek1", 23, "Noida")
print(g.getName(), g.getAge(), g.getAddress())

"""
O/P:
Geek1 23 Noida
"""

"""
Private members of parent class 
We don’t always want the instance variables of the parent class to be inherited by the child class i.e. we can make some of the instance variables of the parent class private, which won’t be available to the child class. 
We can make an instance variable by adding double underscores before its name.
"""


# Private members
class C(object):
    def __init__(self):
        self.c = 21

        # d is private instance variable
        self.__d = 42

    # private method
    def __getd(self):
        return self.__d

class D(C):
    def __init__(self):
        self.e = 84
        C.__init__(self)


object1 = D()

# produces an error as d is private instance variable
# print(object1.d)  # AttributeError: 'D' object has no attribute 'd'
