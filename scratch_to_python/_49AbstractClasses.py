"""
https://www.geeksforgeeks.org/abstract-classes-in-python/?ref=lbp

An abstract class can be considered as a blueprint for other classes. It allows you to create a set of methods that must be created within any child classes built from the abstract class. A class which contains one or more abstract methods is called an abstract class. An abstract method is a method that has a declaration but does not have an implementation. While we are designing large functional units we use an abstract class. When we want to provide a common interface for different implementations of a component, we use an abstract class.

Why use Abstract Base Classes :
By defining an abstract base class, you can define a common Application Program Interface(API) for a set of subclasses. This capability is especially useful in situations where a third-party is going to provide implementations, such as with plugins, but can also help you when working in a large team or with a large code-base where keeping all classes in your mind is difficult or not possible.

How Abstract Base classes work :
By default, Python does not provide abstract classes.
Python comes with a module that provides the base for defining Abstract Base classes(ABC) and that module name is ABC.
ABC works by decorating methods of the base class as abstract and then registering concrete classes as implementations of the abstract base. A method becomes abstract when decorated with the keyword @abstractmethod.
"""

# Python program showing
# abstract base class work

from abc import ABC, abstractmethod


class Polygon(ABC):

    @abstractmethod  # As this class has an abstract method, you can't instantiate i.
    def noofsides(self):
        pass

    @abstractmethod  # Basically, this annotation forces you to override this method in derived classes. Unlike java, python doesn't stop you from having body inside abstract method.
    def enjoy(self):
        print("enjoy")


class Triangle(Polygon):

    # overriding abstract method
    # If you don't override this method, you will get an error
    # TypeError: Can't instantiate abstract class Triangle with abstract method noofsides
    def noofsides(self):
        print("I have 3 sides")

    def enjoy(self):
        super().enjoy()
        print("enjoy drawing Triangle")


class Pentagon(Polygon):

    # overriding abstract method
    def noofsides(self):
        print("I have 5 sides")

    def enjoy(self):
        super().enjoy()
        print("enjoy drawing Pentagon")


class Hexagon(Polygon):

    # overriding abstract method
    def noofsides(self):
        print("I have 6 sides")

    def enjoy(self):
        super().enjoy()
        print("enjoy drawing Hexagon")


class Quadrilateral(Polygon):

    # overriding abstract method
    def noofsides(self):
        print("I have 4 sides")

    def enjoy(self):
        super().enjoy()
        print("enjoy drawing Quadrilateral")


# Driver code
# P = Polygon()  # TypeError: Can't instantiate abstract class Polygon with abstract methods enjoy, noofsides

R = Triangle()
R.noofsides()  # I have 3 sides
R.enjoy()  # enjoy      enjoy drawing Triangle

K = Quadrilateral()
K.noofsides()  # I have 4 sides
K.enjoy()  # enjoy      enjoy drawing Quadrilateral

R = Pentagon()
R.noofsides()  # I have 5 sides
R.enjoy()  # enjoy      enjoy drawing Pentagon

K = Hexagon()
K.noofsides()  # I have 6 sides
K.enjoy()  # enjoy      enjoy drawing Hexagon
