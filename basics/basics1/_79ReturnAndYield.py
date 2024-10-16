"""
https://www.geeksforgeeks.org/use-yield-keyword-instead-return-keyword-python/
https://www.geeksforgeeks.org/generators-in-python/

The yield statement suspends function’s execution and sends a value back to the caller, but retains enough state to enable function to resume where it is left off. When resumed, the function continues execution immediately after the last yield run.
This allows its code to produce a series of values over time, rather than computing them at once and sending them back like a list.
"""

# A generator function that yields 1 for the first time,
# 2 second time and 3 third time
def simpleGeneratorFun():
    # print("hi 1")
    yield 1
    # print("hi 2")
    yield 2
    # print("hi 3")
    yield 3
    # print("hi 4")

# Driver code to check above generator function
v = simpleGeneratorFun()  # generator object
# print(v)  # <generator object simpleGeneratorFun at 0x000002B878215EE0>

"""
# Looping through generator
for value in v:
    print(value)
"""

# Iterating over the generator object using next
print(v.__next__()) # 1
print(v.__next__()) # 2
print(v.__next__()) # 3


print("---------------------------------------------------")


# Return keyword
def fun():
    S = 0

    for i in range(10):
        S += i
    return S


print(fun())


# Yield Keyword
def fun():
    S = 0

    for i in range(10):
        S += i
        yield S


v = fun()
print(v)
print()
for i in v:
    print(i, end=", ")
