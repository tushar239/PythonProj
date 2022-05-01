"""
Addition - X+Y
Subtraction - X-Y
Multiplication - X*Y
Division - X/Y
Modulus - X%Y
Round - round(X)
Boolean or - X or Y
Boolean and - X and Y
Boolean not - not X
Equal comparison - X == Y
Greater than comparison - X > Y
Less than comparison - X < Y
String concatenation - 'Hello' + 'World'
                     - 1 + 2 = 3
                     - '1' + 2 = 12 - TypeError: unsupported operand type(s) for +: 'int' and 'str'
Length of String - len('world')
Character in string - 'world'[0] = w
"""

# Examples: https://www.geeksforgeeks.org/python-logical-operators-with-examples-improvement-needed/

x = int()  # this will assign 0 to x.
x = 5  # You don't declare identifiers with types in Python, that's what dynamic typing means.
y: int = 10  # you don't have to assign a type to any variable. Even though, you assign, there is no strict typing in Python. You can assign string value to the same variable later on.
z = x + y
print(z)

"""
y = 'World'
z = x + y  # TypeError: unsupported operand type(s) for +: 'int' and 'str'
print(z)
"""

z = (1 < 2) or (4 > 5)
print(z)  # true

z = (1 < 2) and (4 > 5)
print(z)  # false

# if x == True: # doesn't work in python
if x: # In python, 0 means false and non-zero is true
    print("x is true")

a = 9
b = 3
if (a % b == 0):
    print("a can be divided by b")


