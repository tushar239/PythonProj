"""
Input Function
1. Input is taken in some variable.
2. Python takes input as a string by default
"""

age = input("What's your age?")  # input function is like an 'answer' block in Scratch. It returns string.
print(type(age))  # <class 'str'>
print(age)

age = int(age)
print(type(age))  # <class 'int'>
print(age)
