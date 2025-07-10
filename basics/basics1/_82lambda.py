import pandas as pd
'''
lambda args: single_expression
'''

# 1. Using lambda with map()
numbers = [1, 2, 3, 4, 5]
# Square each number
squares = list(map(lambda x: x ** 2, numbers))
print(squares) # [1, 4, 9, 16, 25]

# 2. Using lambda with filter()
# Keep even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens) # [2, 4]

# 3. List Comprehension (Alternative to Lambda)
# This is more Pythonic than map() in many cases.
# Double each number
doubled = [x * 2 for x in numbers]
print(doubled)

# 4. Lambda on a List of Tuples
pairs = [(1, 2), (3, 4), (5, 6)]

# Swap elements in each tuple
swapped = list(map(lambda x: (x[1], x[0]), pairs))
print(swapped) # [(2, 1), (4, 3), (6, 5)]

#  5. Using Lambda in Sorting a List
names = ['apple', 'banana', 'cherry']

# Sort by length of string
sorted_names = sorted(names, key=lambda x: len(x))
print(sorted_names) # ['apple', 'banana', 'cherry']










###################################################################
'''
A Python lambda function is limited to a single expression,
but you can handle multiple operations using a few clever techniques.
'''

# lambda x: print(x); x**2  ❌  # SyntaxError! Multiple expressions are not allowed

# But here are ways to simulate multiple expressions in a lambda:

# 1. Use a Tuple to Evaluate Multiple Expressions
f = lambda x: (print(f"Input: {x}"), x ** 2)[1]  # Use tuple and only return the second expression
print(f(5))  # Output: Input: 5 \n 25
# print(...) runs first (expression 0), x**2 runs second (expression 1)
# We return the second by indexing [1]

f = lambda x: (print(f"Input: {x}"), x ** 2)
t = f(5)
print(t[1]) # Output: Input: 5 \n 25


# 2. Use a Function Instead for Clarity
# If you're doing more than one thing, prefer a named function:
def process(x):
    print(f"Processing: {x}")
    return x ** 2

print(process(5))  # Output: Processing: 5 \n 25


#  3. Lambda Returning Multiple Values

df = pd.DataFrame({'value': [5, 10, 15]})

# Lambda returns a tuple of (square, cube)
# df[['square', 'cube']] = df['value'].apply(lambda x: (x**2, x**3)).apply(pd.Series)
df[['square', 'cube']] = df['value'].apply(lambda x: (x**2, x**3)).apply(lambda t : pd.Series(t))
print(df)
'''
   value  square  cube
0      5      25   125
1     10     100  1000
2     15     225  3375

'''
