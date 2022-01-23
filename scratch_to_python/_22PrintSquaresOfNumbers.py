"""
Square of any number is n * n
"""

i = 1

while i < 6:
    # printing in one line with comma separated statements
    print(i*i, end=",")  # to print a number series, you can do print(i)
    i = i+1

print()

# you can use a for loop also
for i in range(1, 6):  # range(1, 6) is same as range(1, 6, 1)
    print(i * i, end=",")

print()

# print(range(1, 6).__contains__(5))
# print(range(1, 6).__ne__(range(1, 6)))
