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
for i in range(1, 6):
    print(i * i, end=",")
