"""
The def keyword is used to define a function in Python, as we have seen in the previous chapter.
The lambda keyword is used to define ANONYMOUS functions in Python.
Usually, such a function is meant for one-time use.

lambda [arguments] : expression

The lambda function can have zero or more arguments after the : symbol.
When this function is called, the expression after : is executed.

The lambda function can have only one expression. Obviously, it cannot substitute a function whose body may have conditionals, loops, etc.

Why Use Lambda Functions?
The power of lambda is better shown when you use them as an anonymous function inside another function.
Use lambda functions when an anonymous function is required for a short period of time.
"""

def outer(num1):
    return lambda num2: num1 + num2


inner = outer(11)

result = inner(33)
print(result)  # 44
result = inner(44)
print(result)  # 55
result = inner(55)
print(result) # 66

greet = lambda name: print('Hello ', name)
print(greet("Steve"))  # Hello Steve

sum = lambda x, y, z : x + y + z
sum(5, 10, 15)  # 30

# The lambda function can have only one expression. Obviously, it cannot substitute a function whose body may have conditionals, loops, etc.
# greet = lambda name: for i in "Steve": print('Hello ', i)

sum = lambda *x: x[0]+x[1]+x[2]+x[3]
result = sum(5, 10, 15, 20)
print(result)  # 50
