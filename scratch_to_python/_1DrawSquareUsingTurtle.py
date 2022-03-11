# turtle is a library in python. Basically, it is just like a sprite in scratch. This sprite is the shape of cursor
import turtle

turtle.color('red', 'yellow')  # setting pen color and fill color

# draw a square using turtle
# to draw a square (4 sides, each with 90 degrees), you just need to draw lines with the angles of 360/4=90

turtle.begin_fill()

turtle.forward(50)  # forward/fd and back/bk methods are like a 'move' block in scratch
turtle.left(90)  # left and right methods are like 'turn' blocks in scratch
turtle.forward(50)
turtle.left(90)
turtle.forward(50)
turtle.left(90)
turtle.forward(50)
turtle.end_fill()


# this code will move turtle to a specific position (x,y coordinates) without drawing a line.
# That's why you need to use penup() first.
turtle.penup()  # penup() or pu() or up(). It is like 'pen up' block in scratch
pos = turtle.pos()  # This method returns a tuple that contains x and y positions of turtle. tuple is a list of values. you can access these values using index.
turtle.goto(pos[0], pos[1] - 50)  # moving 50 pixels down
turtle.pendown()  # pendown() or pd() or down()

# Drawing Pentagon
# 360/5 = 72. after drawing a line, you should take an angle of 72 degrees.

turtle.fillcolor('green')

turtle.begin_fill()

turtle.forward(50)
turtle.left(72)
turtle.forward(50)
turtle.left(72)
turtle.forward(50)
turtle.left(72)
turtle.forward(50)
turtle.left(72)
turtle.forward(50)


turtle.end_fill()

# this won't close the window automatically.
turtle.done()
