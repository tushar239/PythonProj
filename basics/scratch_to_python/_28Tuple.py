"""
Tuple = immutable list
For Tuple, use () instead of []. [] is for list.

So, basically when you want a function to return something that can't be changed, it can return Tuple instead of List.
"""

t1 = (3, 4, 7)
print(t1[0])  # 3

# t1[1] = 45  # error - you can't modify/add anything in Tuple

print(type(t1))  # <class 'tuple'>
del t1  # Any variable can be deleted in this way
# print(type(t1))  # NameError: name 't1' is not defined


