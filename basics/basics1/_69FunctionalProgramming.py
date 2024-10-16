"""
https://www.geeksforgeeks.org/functional-programming-in-python/?ref=lbp

Concepts of Functional Programming

Pure Functions
    Pure functions have two properties.

    - It always produces the same output for the same arguments. For example, 3+7 will always be 10 no matter what.
    - It does not change or modifies the input variable.

    The second property is also known as immutability. The only result of the Pure Function is the value it returns. They are deterministic.
    Programs done using functional programming are easy to debug because pure functions have no side effects or hidden I/O.
    Pure functions also make it easier to write parallel/concurrent applications.
    When the code is written in this style, a smart compiler can do many things – it can parallelize the instructions, wait to evaluate results when needing them, and memorize the results since the results never change as long as the input doesn’t change.

Recursion
    During functional programming, there is no concept of for loop or while loop, instead recursion is used.
    Recursion is a process in which a function calls itself directly or indirectly.
    In the recursive program, the solution to the base case is provided and the solution to the bigger problem is expressed in terms of smaller problems.
    A question may arise what is base case? The base case can be considered as a condition that tells the compiler or interpreter to exit from the function.

Functions are First-Class and can be Higher-Order

    - A function is an instance of the Object type.
    - You can store the function in a variable.
    - You can pass the function as a parameter to another function.
    - You can return the function from a function.
    - You can store them in data structures such as hash tables, lists etc.

    Functions are called Higher-Order, if
    - you can return the function from a function
    or
    - you can pass function as a parameter to another function
"""

# Example of pure function on object-oriented world
def pure_func(alist):
    new_list = []

    for i in alist:
        new_list.append(i ** 2)

    return new_list


Original_List = [1, 2, 3, 4]
Modified_List = pure_func(Original_List)

print("Original List:", Original_List)  # [1, 2, 3, 4]
print("Modified List:", Modified_List)  # [1, 4, 9, 16]

# Example of Recursive function to find a sum of a list
def listsum(alist, start, end):
    if start == end:
        return alist[start]

    res = alist[start] + listsum(alist, start+1, end)
    return res

alist = [1, 2, 3, 4]
result = listsum(alist, 0, len(alist)-1)
print("Sum using recursion:", result)

# Example of function as a First class citizen

def shout(text):
    return text.upper()


def whisper(text):
    return text.lower()


def greet(func):
    # storing the function in a variable
    greeting = func("Hi, I am created by a function passed as an argument.")
    print(greeting)


greet(shout)  # HI, I AM CREATED BY A FUNCTION PASSED AS AN ARGUMENT.
greet(whisper)  # hi, i am created by a function passed as an argument.

# Built-in higher-order functions

def addition(n):
    return n + n

# We double all numbers using map()
numbers = (1, 2, 3, 4)
results = map(addition, numbers)  # map class inherits Iterator. When results is iterated, at that time, logic (addition function) is applied on input (numbers).

# Does not Print the value
print(results)  # <map object at 0x000002E408877E50> - it's a map class object

# When results is iterated, at that time, logic (addition function) is applied on input (numbers).
for result in results:
    print(result, end=" ")  # 2 4 6 8

print()

# Example of 'filter'
def fun(variable):
    letters = ['a', 'e', 'i', 'o', 'u']

    if variable in letters:
        return True
    else:
        return False


# sequence
sequence = ['g', 'e', 'e', 'j', 'k', 's', 'p', 'r']

# using filter function
filtered = filter(fun, sequence)  # filtered is an object of filter class. filter class inherits Iterator.

print('The filtered letters are:')

# When results is iterated, at that time, logic (fun function) is applied on input (sequence).
for s in filtered:
    print(s, end=" ")    # e e

print()
