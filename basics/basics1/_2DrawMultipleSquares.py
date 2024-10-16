# Teach this code without using function and loops. Teach this code again after teaching 'Controls'.

import turtle
import time

# introducing a function
def drawsquare():
    turtle.pendown()

    turtle.color('red', 'purple')

    turtle.begin_fill()

    turtle.forward(50)
    turtle.right(90)
    turtle.forward(50)
    turtle.right(90)
    turtle.forward(50)
    turtle.right(90)
    turtle.forward(50)
    turtle.right(90)

    turtle.end_fill()

    turtle.penup()


turtle.setpos(0, 0)


for num in range(0, 4):  # this is like for(int i=0; i<4; i++) in java
    drawsquare()
    time.sleep(1)
    # To draw a second square, turn left by 90 degrees
    turtle.left(90)


turtle.done()


