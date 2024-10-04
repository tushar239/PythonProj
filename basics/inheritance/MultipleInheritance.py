# https://www.geeksforgeeks.org/multiple-inheritance-in-python/

class Class1:

    def m(self):
        print("In Class1")


class Class2(Class1):
    def m(self):
        pass


class Class3(Class1):
    def m(self):
        print("In Class3")


# Multiple Inheritance

class Class4(Class2, Class3):
    pass


# If both Class2 and Class3 have m(), then whichever class is mentioned first in order, that class' m() will be called.
obj = Class4()
obj.m()
