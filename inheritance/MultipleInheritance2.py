# https://www.geeksforgeeks.org/multiple-inheritance-in-python/

# when every class defines the same method

class Class1:
    def m(self):
        print("In Class1")


class Class2(Class1):
    def m(self):
        print("In Class2")


class Class3(Class1):
    def m(self):
        print("In Class3")


class Class4(Class2, Class3):
    def m(self):
        print("In Class4")


obj = Class4()
obj.m()

Class2.m(obj)
Class3.m(obj)
Class1.m(obj)

'''
O/P:
In Class4
In Class2
In Class3
In Class1
'''