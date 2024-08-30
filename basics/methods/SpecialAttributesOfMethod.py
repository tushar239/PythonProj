"""
https://docs.python.org/3/library/stdtypes.html#other-built-in-types

Every instance method (Bound Method) has attributes called __self__ and __func__.
__self__ attribute stores the reference of a class instance.
__func__ attribute is the function implementing the method.
    Calling m(arg-1, arg-2, ..., arg-n) is completely equivalent to calling m.__func__(m.__self__, arg-1, arg-2, ..., arg-n).
    You can add attributes to __func__ like method.__func__.hi = "hi"

"""


class C:
    def __repr__(self, *args, **kwargs):
        return super().__repr__(*args, **kwargs)

    def method(self, name):
        "I am a method"
        self.name=name
        print(name)
        pass
    def method1(self):
        print("Inside method1")

c=C()
print(repr(c)) # <__main__.C object at 0x000001D6746FC810>
print(str(c)) # <__main__.C object at 0x000001D6746FC810>
print(type(c)) # <class '__main__.C'>
c.method("Tushar")
print(type(c.name)) # <class 'str'>

print(c.method.__self__) # <__main__.C object at 0x000001D6746FC810>
c1 = c.method.__self__
c1.method1() # Inside method1

print(c.method.__func__) # <function C.method at 0x000001D6746E96C0>
print(c.method.__func__(c.method.__self__,"Tushar"))

print(c.method.__str__()) # <bound method C.method of <__main__.C object at 0x102078908>>
print(c.method.__module__) # __main__

#c.method.whoami  = 'my name is method'  # can't set on the method
c.method.__func__.whoami = 'my name is method'
c.method.__func__.hi = "hi"
print(c.method.__func__.whoami) # my name is method
print(c.method.__func__.hi) # hi

# Referencing class attributes
m1=c.method1
m1() # Inside method1

i = int(x="10")
print(type(i)) # <class 'int'>

j = 10
print(type(j)) # <class 'int'>
print(i == j)

if __name__ == '__main__':
    print("yes")