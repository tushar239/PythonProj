# Python program to demonstrate
# super()

class Class1:
    def m(self):
        print("In Class1")


class Class2(Class1):
    def m(self):
        print("In Class2")
        super().m()


class Class3(Class1):
    def m(self):
        print("In Class3")
        super().m()


class Class4(Class2, Class3):
    def m(self):
        print("In Class4")
        super().m()


obj = Class4()
obj.m()

# MRO is called Method Resolution Order (MRO).
print(Class4.mro())         # This will print list
print(Class4.__mro__)        # This will print tuple

'''
In Class4
In Class2
In Class3
In Class1

[<class '__main__.Class4'>, <class '__main__.Class2'>, <class '__main__.Class3'>, <class '__main__.Class1'>, <class 'object'>]
(<class '__main__.Class4'>, <class '__main__.Class2'>, <class '__main__.Class3'>, <class '__main__.Class1'>, <class 'object'>)
'''
# __main__ is the name of the environment where top-level code is run. “Top-level code” is the first user-specified
# Python module that starts running. It's “top-level” because it imports all other modules that the program needs.
# Sometimes “top-level code” is called an entry point to the application.