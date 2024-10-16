"""
In Python, you can calculate the quotient with // and the remainder with % .
The built-in function divmod() is useful when you want both the quotient and the remainder.
divmod(a, b) returns a tuple (a // b, a % b) .
"""

a = 10
b = 3

print("Quotient: ", a // b)
print("Remainder: ", a % b)

quotientandremainder = divmod(a, b)

print("Quotient: ", quotientandremainder[0])
print("Remainder: ", quotientandremainder[1])
