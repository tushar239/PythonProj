
b = "Hello, World!"
print(b[2:5])  # llo
print(b[:5])  # Hello
print(b[2:])  # llo, World!
print(b[-5:-2])  # orl  --- -2,-3,-4 index elements are considered

print("apple".index("apple"))  # 0 --- if not found, then error will be thrown
print("apple".find("pple"))  # 1
print("apple"[1:3])  # pp