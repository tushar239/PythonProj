# https://www.geeksforgeeks.org/g-fact-41-multiple-return-values-in-python/

# A Python program to return multiple
# values from a method using tuple

# This function returns a tuple
def fun():
    str = "geeksforgeeks"
    x = 20
    return str, x  # Return tuple, we could also
    # write (str, x)


# Driver code to test above method
str, x = fun()  # Assign returned tuple
print(str)
print(x)


# A Python program to return multiple
# values from a method using class
class Test:
    def __init__(self):
        self.str = "geeksforgeeks"
        self.x = 20


# This function returns an object of Test
def fun():
    return Test()


t = fun()
print(t.str)
print(t.x)


# A Python program to return multiple
# values from a method using list

# This function returns a list
def fun():
    str = "geeksforgeeks"
    x = 20
    return [str, x]


list = fun()
print(list)


# A Python program to return multiple
# values from a method using dictionary

# This function returns a dictionary
def fun():
    d = dict();
    d['str'] = "GeeksforGeeks"
    d['x'] = 20
    return d


d = fun()
print(d)