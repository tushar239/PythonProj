"""
See the video 'Object Oriented Programming, Classes, Objects.mp4'.

https://www.w3schools.com/python/python_classes.asp
https://www.tutorialsteacher.com/python/python-class

Python Classes/Objects
Python is an object oriented programming language.

Almost everything in Python is an object, with its properties and methods.

A Class is like an object constructor, or a "blueprint" for creating objects

Python is a completely object-oriented language. You have been working with classes and objects right from the beginning of these tutorials. Every element in a Python program is an object of a class. A number, string, list, dictionary, etc., used in a program is an object of a corresponding built-in class. You can retrieve the class name of variables or objects using the type() method, as shown below.

Example: Python Built-in Classes Copy
     num=20
     type(num)  # <class 'int'>
     s="Python"
     type(s)  # <class 'str'>

A class in Python can be defined using the class keyword.

class <ClassName>:
    <statement1>
    <statement2>
    .
    .
    <statementN>

As per the syntax above, a class is defined using the class keyword followed by the class name and : operator after the class name, which allows you to continue in the next indented line to define class members. The followings are class members.

Class Attributes
Constructor
Instance Attributes
Properties
Class Methods
"""
# sometimes, you just want to create an empty class for some kind of placeholder. Just like function, you can't have empty class. But you can just add 'pass' statement.
class MyClass:
    pass

class MyClass1:
    x = 5

obj = MyClass1()
print(obj.x)

# A magic method __init__() is like a constructor in Python.
# The __init__() function is called automatically every time the class is being used to create a new object.
class Person:
    def __init__(self, name, age):  # self - it means current instance. It is like 'this' in java. You can use any variable name like 'abc', 'myinstance' etc instead of 'self'
        self.name = name
        self.age = age

    def myfunc(self):
        print("Hello my name is " + self.name)

person = Person("John", 36)
print(person.name,person.age, sep=", ")  # John, 36
person.myfunc()  # Hello my name is John
