import turtle


# introducing a function
def drawsquare(pencolor):
    turtle.pendown()

    turtle.pencolor(pencolor)

    turtle.forward(50)
    turtle.right(90)
    turtle.forward(50)
    turtle.right(90)
    turtle.forward(50)
    turtle.right(90)
    turtle.forward(50)
    turtle.right(90)

    turtle.penup()


turtle.setpos(0, 0)  # setting initial position of a turtle

drawsquare('red')
# To draw a second square, turn left by 90 degrees
turtle.left(90)
drawsquare('blue')
turtle.left(90)
drawsquare('yellow')
turtle.left(90)
drawsquare('green')
turtle.left(90)

turtle.done()
