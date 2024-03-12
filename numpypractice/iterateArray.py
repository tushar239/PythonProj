import numpy as np

array = np.arange(start=1, stop=5, step=.5)

# This is just like a smart for loop in java
for num in array:
    print(num)

print("--------------------")

# this is like normal for loop in java
for i in np.arange(start=0, stop=array.size, step=1, dtype=int):
    print(str(array[i]))

print("--------------------")

# range is a class. so, you can't do range(start=0, stop=array.size, step=1
indices = range(0, array.size, 1)
for i in indices:
    print(str(array[i]))

print("--------------------")