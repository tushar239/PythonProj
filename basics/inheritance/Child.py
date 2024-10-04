# object is top base class and if you want to use super(...).method(...) calling feature, then you have to extend
# object like this or use '__metaclass__ = type'. I don't know why ???
from basics.inheritance.Parent import Parent


# you don't need it for Python 3.x
# class Child(Parent, object)
class Child(Parent):  # This is same as 'Child extends Parent' in java.

    def __init__(self):
        super().__init__() # equivalent to super( Child, self).__init__()
        self.childVar = Parent.parentStaticVar

    # In Python method overloading is not allowed, but method overriding is allowed. There is a reason why it is not
    # allowed. Python supports assigning default values to method parameters. So that is equivalent to method
    # overloading because client can pass different number of parameters to the method by ignoring parameters which
    # are assigned default values. Overridden method from Parent class
    def method1(self, var):
        self.parentVar = var
        print("Inside Child->method1() : " + str(var))

    def get_child_var(self):
        return self.childVar

    # calling super's method
    # super(type) -> unbound super object  --- ???
    # super(type, obj) -> bound super object; requires isinstance(obj, type)
    # super(type, type2) -> bound super object; requires issubclass(type2, type) --- ???
    def method2(self, var):
        super(Child, self).method1(var)

    # Right Click -> Generate -> Override Methods
    # OR
    # Menu -> Code -> Generate -> Override Methods

    # __del__ method is called when obj is going be garbage collected
    def __del__(self):
        print(self.__class__.__name__ + " instance is going to be garbage collected")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Child) and self.childVar == other.childVar

    # OR __eq__ and __cmp__ are same
    '''
    def __cmp__(self, other):
        return isinstance(other, Child) and self.childVar == other.childVar
    '''

    # It is like hashCode() method in Java
    def __hash__(self):
        # hash(custom_object)
        return hash((self.parentVar, self.childVar))

    # It is like toString() method in Java
    def __str__(self):
        return "Inside Child->__str__(), childVar:" + str(self.childVar)

    # __repr__() provides representation of an object
    # When __str__() is not available, Python will call __repr__() for print(obj)
    # But that is not the only use of __repr__().
    # repr(obj) calls obj.__repr__. It is mainly used for debugging purpose by programmers.
    # For actual user experience, you should use __str__()

    # http://www2.lib.uchicago.edu/keith/courses/python/class/5/#__repr__ The __str__ method is exactly like __repr__
    # except that it is called when the builtin str function is applied to an object; this function is also called
    # for the %s escape of the % operator. In general, the string returned by __str__ is meant for the user of an
    # application to see, while the string returned by __repr__ is meant for the programmer to see, as in debugging
    # and development: but there are no hard and fast rules about this. You're best off just thinking, __str__ for
    # %s, __repr__ for backquotes.
    def __repr__(self):
        return "Inside Child->__repr__(), childVar:" + str(self.childVar)


# Python repr() function returns a printable representation of the object by converting that object to a string.
print("The value of __name__ is:", repr(__name__))

if __name__ == "__main__":
    child = Child()
    child.method1(15)
    print("childVar: " + str(child.get_child_var()))  # 10

    # issubclass(subclass, superclass)
    print("is Child a subclass of Parent class: " + str(issubclass(Child, Parent)))  # True
    # isinstance(obj, Class)
    print("is child an instance of Child class: " + str(isinstance(child, Child)))  # True
    print("is child an instance of Parent class: " + str(isinstance(child, Parent)))  # True
    print(child)  # It calls __str__ method, just like Java
    print(repr(child))  # It calls __repr__ method, just like Java
    print("Testing super method call: " + str(child.method2(10)))
    print("Testing static method call: " + str(Child.get_parent_static_var()))  # 10

    child2 = Child()
    print("is child and child2 equals: " + str(child.__eq__(child2)))  # True
    # del child
