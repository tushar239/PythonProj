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

Magic Methods:
https://www.analyticsvidhya.com/blog/2021/08/explore-the-magic-methods-in-python/
https://www.tutorialsteacher.com/python/magic-methods-in-python

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
    # self - it means current instance of the class and it is used to access variables that belong to the class.
    # It is like 'this' in java.
    # It does not have to be named self , you can call it whatever you like, but it has to be the first parameter of any function in the class.
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def myfunc(self):
        print("Hello my name is " + self.name)

    # This is a magic method. Python has many inbuilt magic methods. They work as operator overloading.
    # Whenever you do instance1 + instance2, __add__ method will be called
    def __add__(self, other):
        return self.age + other.age

    # This is also a magic method. When you compare two instances of this class (instance1 == instance2), this magic method will be called.
    def __eq__(self, other):
        return self.age == other.age

    def __hash__(self):
        return hash(self.age)  # not doing any changes

    def __str__(self):
        # Using Turnary operator
        # Syntax: (if_test_is_false, if_test_is_true)[test]
        return "#".join([self.name,
                         ("", str(self.age))[self.age != None]  # Using Ternary operator
                         ])
        """
        return "#".join([self.name, 
                            str(self.age) if self.age != None else ""  # Using Comprehension
                        ])
        """
        # return self.name+"#",self.age


person = Person("John", 36)
print(person.name, person.age, sep=", ")  # John, 36
person.myfunc()  # Hello my name is John

# Modify property. When you modify a property of one instance, it doesn't affect another instance.
person.age = 37

person2 = Person("Steve", 35)
print(person + person2)  # 37 + 35 = 72

person.age = 35
print(person == person2)  # True

person.age = 37
print(person == person2)  # False

person.age = 35
s = {person, person2}  # as hash of both instances are same, set will keep only one of these instances.
for element in s:
    print(element)  # John#35 --- when you print an instance, __str__ method will be called.

# deleting an instance. This is like person=null in java.
del person
# deleting a property of an instance. This is like person2.age=null in java.
del person2.age
