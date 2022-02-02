"""
This program is an example of how to use if-else if-else
Next program will show you another way of writing the same program
"""
import random

randomnumber = random.randint(2, 7)

biggest = None

for index in range(0, randomnumber):
    num = int(input("Enter number: "))
    if biggest is None:      #   The Equality operator (==) compares the values of both the operands and checks for value equality. Whereas the 'is' operator checks whether both the operands refer to the same object or not (present in the same memory location)
        biggest = num
    elif num > biggest:
        biggest = num

print("Biggest number is ", biggest)
