"""
String is immutable. When you try to apply any function that modifies the string, it will return a new string object.
Remember: In Python, everything is Class, there are no literals.
"""
a = "Hello, World!"
print(a.upper())  # HELLO, WORLD!

a = "Hello, World!"
print(a.lower())  # hello, world!

a = " Hello, World! "
print(a.strip())  # Hello, World! --- it is same as trim() in Java

a = "Hello, World!"
print(a.replace("o", "J"))  # HellJ, WJrld!

a = "Hello, World!"
print(a.split(","))  # ['Hello', ' World!']
