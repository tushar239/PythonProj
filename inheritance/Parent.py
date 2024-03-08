class Parent:  # by default it extends 'object' class from __builtins__ module.
    parentStaticVar = 10  # it's a static variable as it is not like self.parentStaticVar.

    def __init__(self):  # This is like a initialization method which is called after constructor (__new__(self) method) by Python. In Python program, you generally don't define a constructor, but you define init method.
        self.parentVar = 10  # when you do self.var, var becomes an instance variable.

    def method1(self, var):  # non-static method takes 'self' as first parameter
        self.parentVar = var
        print("Inside Parent->method1() : " + str(var))

    def get_parent_var(self):
        return self.parentVar

    @classmethod  # you can use a decorator @staticmethod also to declare a method as static.
    def get_parent_static_var(self):  # 'self' is not mandatory for static methods
        # print(self)
        return Parent.parentStaticVar  # This is how you access static var/method

    def __str__(self):  # it is like toString() in java. __str__ is a method inside 'object' class of __builtins__ module and you are overriding it it here.
        return "Inside Parent->__str__(), childVar:" + str(self.parentVar)
