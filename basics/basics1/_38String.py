"""
Strings in python are surrounded by either single quotation marks, or double quotation marks.

'hello' is the same as "hello".

There is no Data Type called Character in Python. So, string can be represented either in single quotes also.

Like many other popular programming languages, strings in Python are arrays of bytes representing unicode characters.

However, Python does not have a character data type, a single character is simply a string with a length of 1.

Square brackets can be used to access elements of the string.

String in IMMUTABLE in Python like Java and many other programming languages.
https://www.tutorialspoint.com/python_text_processing/python_string_immutability.htm
"""

a = "Hello"
print(a)  # Hello

# Immutability - Can not reassign - https://www.tutorialspoint.com/python_text_processing/python_string_immutability.htm
t = "Tutorialspoint"
print(type(t))
# t[0] = "M" # TypeError: 'str' object does not support item assignment

# When we run the above program we get the following output.
# As you can see above a and a point to same location. Also N and N also point to the same location.
x = 'banana'
for idx in range(0, len(x)):
    print(x[idx], "=", id(x[idx]))

# Multiline quotes --- you can use three double quotes or three single quotes
a = """Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua."""
print(a)
"""
Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua.
"""

# Accessing characters of string
a = "Hello, World!"
print(a[1])
# a[1] = 'I'  # TypeError: 'str' object does not support item assignment. --- String is immutable

# String is an array in itself, so you can access string characters just like looping an array
for x in "banana":
    print(x, end=", ")

print()  # b, a, n, a, n, a,

a = "Hello, World!"
print(len(a))  # 13

# Check if "free" is present in the text
txt = "The best things in life are free!"
print("free" in txt)

# Print only if "free" is present
txt = "The best things in life are free!"
if "free" in txt:  # implicitly calls a magic method __contains__ of str class. Same as subString() of Java
    print("Yes, 'free' is present.")  # Yes, 'free' is present.

# Check if "expensive" is NOT present in the following text:
txt = "The best things in life are free!"
if "expensive" not in txt:  # implicitly calls a magic method __contains__ of str class
    print("expensive not in txt")  # expensive not in txt

# OR
if txt.find("expensive") == -1:
    print("expensive not in txt")  # expensive not in txt

# print(txt.index("expensive")) # Java returns -1, if substring not found in a string, but python throws an error because -1 is an index in Python(last char of string)
# You can use txt.find("expensive"), if you want -1 as returned value, in case a substring is not found in a string.

s: str = "abcad"
count: int = 0
while "a" in s: # see whether 'a' is inside
    itr = s.find("a")
    count = count + 1
    # s.replace('a', '') # all a's will be replaced
    s.replace('a', '', 1) # replaces only first available a
print("total number of a's "+count)


# ord() function to get a number for a character
# https://www.w3schools.com/python/ref_func_ord.asp
x = ord("h") # 104
