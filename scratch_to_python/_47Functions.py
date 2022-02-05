# In Python a function is defined using the 'def' keyword
def my_function():
    print("Hello from a function")


my_function()  # Calling a function


# Arguments
def my_function(fname):
    print(fname, "Hosbon")


my_function("John")
my_function("Dev")
my_function("Randall")


def my_function(fname, lname, age):
    print(fname, lname, age, sep=",")


my_function("John", "Hosbon", 40)
my_function(fname="John", age=40,
            lname="Hosbon")  # If you want to send values in a different order than parameters, then you can do it in this way.


# Default value
def my_function(fname, maritalstatus="not married"):
    print(fname, maritalstatus, sep=",")


my_function("John")
my_function("John", "married")


# Arbitrary Arguments (*args) - It's a Tuple
# If you do not know how many arguments that will be passed into your function, add a * before the parameter name in the function definition.
# This way the function will receive a tuple of arguments, and can access the items accordingly
def my_function(fname, *relatives):
    print(fname, relatives)
    print(relatives[1], "is great")


my_function("John", "Dev", "Randall", "Jack")


# OR
def my_function(fname, relatives):
    print(fname, relatives)
    print(relatives[1], "is great")


my_function("John", ("Dev", "Randall", "Jack"))


# Keyword Arguments
# If you do not know how many keyword arguments that will be passed into your function, add two asterisk: ** before the parameter name in the function definition.
# This way the function will receive a dictionary of arguments, and can access the items accordingly.

def my_function(**teachers):
    print("His last name is " + teachers["lname"])


my_function(fname="John", lname="Hosbon")  # His last name is Hosbon


# OR

def hello_function(teachers):
    print("His last name is " + teachers["lname"])


hello_function({"fname": "John", "lname": "Hosbon"})  # His last name is Hosbon


# Passing a list as an Argument
# You can send any data types of argument to a function (string, number, list, dictionary etc.), and it will be treated as the same data type inside the function.

def food_function(food):
    for x in food:
        print(x, end=", ")


fruits = ["apple", "banana", "cherry"]

food_function(fruits)  # apple, banana, cherry,
print()
food_function("hello")  # h, e, l, l, o,
# This can be confusing. When you don't have to define a type of parameter in the function, you have to look into its code to guess the type.
# By looking at food_function(), you can send a list, string, tuple or set, everything will work.
