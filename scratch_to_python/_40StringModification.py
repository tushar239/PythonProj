"""
String is immutable. When you try to apply any function that modifies the string, it will return a new string object.
Remember: In Python, everything is Class, there are no literals.
"""
a = "Hello, World!"
print(a.upper())  # HELLO, WORLD!

a = "Hello, World!"
print(a.lower())  # hello, world!

txt = "Hello My Name Is PETER"
x = txt.swapcase()
print(x)  # hELLO mY nAME iS peter

a = " Hello, World! "
print(a.strip())  # Hello, World! --- it is same as trim() in Java

a = " Hello, World! "
print(a.lstrip())  # 'Hello, World! '
print(a.rstrip())  # ' Hello, World!'

a = "Hello, World!"
print(a.replace("o", "J"))  # HellJ, WJrld!

a = "Hello, World!"
print(a.split(","))  # ['Hello', ' World!']

# Split a string into a list where each line is a list item
txt = "Thank you for the music\nWelcome to the jungle"
x = txt.splitlines()
print(x)  # ['Thank you for the music', 'Welcome to the jungle']

"""
Search for the word "bananas", and return a tuple with three elements:

1 - everything before the "match"
2 - the "match"
3 - everything after the "match"
"""
txt = "I could eat bananas all day"
x = txt.partition("bananas")
print(x)  # ('I could eat ', 'bananas', ' all day')

# Join all items in a tuple into a string, using a hash character as separator
myTuple = ("John", "Peter", "Vicky")
x = "#".join(myTuple)
print(x)  # John#Peter#Vicky

# Make the first letter in each word upper case.
txt = "Welcome to my world"
x = txt.title()
print(x)  # Welcome To My World


# use a dictionary with ascii codes to replace 83 (S) with 80 (P):
mydict = {83:  80}
txt = "Hello Sam!"
print(txt.translate(mydict))  # Hello Pam!

# Print the word "banana", taking up $ of 20 characters, with "banana" in the middle.
txt = "banana"
x = txt.center(20, "$")
print(x)  # $$$$$$$banana$$$$$$$
