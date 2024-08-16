"""
IMP:
There is actually no concept of interface in Python. In Python, abstract class with only abstract methods can be considered as interface.

NOTE: Do not teach this file code. zope package is actually used for component based design and programming. It is not for interface.
    For making interface in Python, you just create an abstract class with all methods as abstract.

https://www.geeksforgeeks.org/python-interface-module/

In object-oriented languages like Python, the interface is a collection of method signatures that should be provided by the implementing class.
Implementing an interface is a way of writing an organized code and achieve abstraction.

The package 'zope.interface' provides an implementation of “object interfaces” for Python.

The package exports two objects, ‘Interface’ and ‘Attribute’ directly. It also exports several helper methods.
It aims to provide stricter semantics and better error messages than Python’s built-in 'abc' module that is used for Abstraction.

Syntax :
class IMyInterface(zope.interface.Interface):
    # methods and attributes

install a package 'zope.interface' using 'pip install zip.interface'


"""

import zope.interface


class MyInterface(zope.interface.Interface):
    x = zope.interface.Attribute("foo")

    def method1(self, x):
        pass

    def method2(self):
        pass


print(type(MyInterface))
print(MyInterface.__module__)
print(MyInterface.__name__)

# get attribute
x = MyInterface['x']
print("Attribute x", x)  # __main__.MyInterface.foo
print(type(x))  # <class 'zope.interface.interface.Attribute'>


@zope.interface.implementer(MyInterface)
class MyClass:
    def method1(self, x):
        return x ** 2

    def method2(self):
        return "foo"


obj = MyClass()
print(isinstance(obj, MyClass))  # True
print(isinstance(obj, MyInterface))  # False
print(obj.method1(3))  # 9
print(obj.method2())  # foo

# ask an interface whether it
# is implemented by a class:
print(MyInterface.implementedBy(MyClass))  # True

# MyClass does not provide
# MyInterface but implements it:
print(MyInterface.providedBy(MyClass))  # False

# ask whether an interface
# is provided by an object:
print(MyInterface.providedBy(obj))  # True

# ask what interfaces are
# implemented by a class:
print(list(zope.interface.implementedBy(MyClass)))  # [<InterfaceClass __main__.MyInterface>]

# ask what interfaces are
# provided by an object:
print(list(zope.interface.providedBy(obj)))  # [<InterfaceClass __main__.MyInterface>]

# class does not provide interface
print(list(zope.interface.providedBy(MyClass)))


class BaseI(zope.interface.Interface):
    def m1(self, x):
        pass

    def m2(self):
        pass


class DerivedI(BaseI):
    def m3(self, x, y):
        pass


@zope.interface.implementer(DerivedI)
class cls:
    def m1(self, z):
        return z ** 3

    def m2(self):
        return 'foo'

    def m3(self, x, y):
        print("inside cls' m3()")
        return x * y * 3


# obj = BaseI()  # you can't instantiate an interface
obj = cls()
print(obj.m3(3, 4))
# Get base interfaces
print(DerivedI.__bases__)  # (<InterfaceClass __main__.BaseI>,)

# Ask whether baseI extends
# DerivedI
print(BaseI.extends(DerivedI))  # False

# Ask whether baseI is equal to
# or is extended by DerivedI
print(BaseI.isEqualOrExtendedBy(DerivedI))  # True

# Ask whether baseI is equal to
# or extends DerivedI
print(BaseI.isOrExtends(DerivedI))  # False

# Ask whether DerivedI is equal
# to or extends BaseI
print(DerivedI.isOrExtends(DerivedI))  # True

from zope.interface import implements


class IPerson(zope.interface.Interface):
    # name = zope.interface.Attribute("Name")
    # email = zope.interface.Attribute("Email Address")
    # phone = zope.interface.Attribute("Phone number")
    pass


@zope.interface.implementer(IPerson)
class Person(object):
    # def __init__(self):
    #   print(IPerson.__getattribute__("name"))
    pass


jack = Person()
# print(jack.email)  # AttributeError: 'Person' object has no attribute 'email'
jack.email = "jack@some.address.com"
print(jack.email)  # jack@some.address.com

steve = Person()
# print(steve.email)  # AttributeError: 'Person' object has no attribute 'email'
