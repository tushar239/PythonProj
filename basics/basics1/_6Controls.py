"""

Videos for kids - https://www.youtube.com/watch?v=eSYeHlwDCNA
                  https://www.youtube.com/watch?v=wxds6MAtUQ0

After teaching controls, teach 'DrawMultipleSquares.py' using loop

if block
--------
Video for kids: https://www.youtube.com/watch?v=w826p9clLeA

if condition:
    execute any statement

if-then-else block
------------------

if condition:
    execute statements, if condition is true
else:
    execute statements, if condition is false


repeat block OR for loop
------------------------

for index in range (10):     value of index will be 0 and go till 9. You can also use range(0,10)
    execute statements

forever OR while True block
---------------------------

while True:                 this will execute statements forever(infinite times)
    execute statements

repeat until OR while not condition block
-----------------------------------------

while not condition:
    execute statements
"""

if 1 > 2:
    print("1 is greater than 2")
else:
    print("1 is less than 2")

num = 0
while True:
    print('Hello ' + str(num))  # you can't concatenate a number with string. To make that possible, you need to convert a number to string and that can be done using a function str(num)
    num = num + 1
    if num > 9:
        break    # helps to come out of the loop

# Now, how to convert above while loop into for loop?

for num in range(10):  # or range(0, 10)
    print('Hello '+str(num))