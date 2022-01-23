result = 0

for i in range(1, 11):
    # sum = sum + i  # TypeError: unsupported operand type(s) for +: 'builtin_function_or_method' and 'int'
    result = result + i

print("sum: ", result)

# Try with while loop also
