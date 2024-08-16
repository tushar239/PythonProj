import turtle

# introducing a function
def drawpentagon():
    turtle.pendown()

    # turtle.color('red', 'purple')

    # turtle.begin_fill()

    turtle.forward(100)
    turtle.right(72)
    turtle.forward(100)
    turtle.right(72)
    turtle.forward(100)
    turtle.right(72)
    turtle.forward(100)
    turtle.right(72)
    turtle.forward(100)
    turtle.right(72)

    # turtle.end_fill()

    turtle.penup()


turtle.setpos(0, 0)


for num in range(0, 12):  # this is like for(int i=0; i<12; i++) in java
    drawpentagon()
    # To draw a second pentagon, turn left by 30 degrees. So, to finish the entire circle, you need to draw a pentagon 360/30=12 times.
    turtle.left(30)


turtle.done()


