"""
In general, you should use a for loop when you know how many times the loop should run.
If you want the loop to break based on a condition other than the number of times it runs, you should use a while loop.

While loop:
1. Initialize
2. condition
3. statements which will repeat
4. incremental statement
"""

i = 1

while i < 3:
    # printing in one line with comma separated statements
    print("hi", end=",")  # to print a number series, you can do print(i)
    i = i+1

print()

# This can be converted into for loop also
for j in range(0, 2):
    print("hi", end=",")

print()

# Following while loop can't be converted into for loop
i = 1
while i > 0:
    i = int(input("Enter a number: "))

print("out of while loop")

while True:
    i = int(input("Enter a number: "))
    if i < 0:
        break
print("out of while loop")

