
# The find() method returns -1 if the value is not found
txt = "Mi casa, su casa."
x = txt.find("casa")
print(x)  # 3

# Where in the text is the last occurrence of the string "casa"?
x = txt.rfind("casa")
print(x)  # 12
x = txt.find("hello")
print(x)  # -1

# index() throws an error, if the value is not found
x = txt.index("casa")
print(x)  # 3
x = txt.rindex("casa")
print(x)  # 12
# txt.index("hello")  # ValueError: substring not found

txt = "Hello, welcome to my world."
x = txt.startswith("Hello")
print(x)  # True
