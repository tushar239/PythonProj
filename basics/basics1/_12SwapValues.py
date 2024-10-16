x = 7
y = 5

print("x=", x)
print("y=", y)

"""
x = y
y = x 
will not work
"""
"""
# generic solution for all programming languages
temp = x
x = y
y = temp

print("x=", x)
print("y=", y)
"""

# python specific solution
x, y = y, x

print("x=", x)
print("y=", y)
