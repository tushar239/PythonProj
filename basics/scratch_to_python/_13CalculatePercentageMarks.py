"""
Input marks of 3 subjects (max marks 100)
calculate %
"""

math = int(input("Enter marks for math (0-100)"))
science = int(input("Enter marks for science (0-100)"))
painting = int(input("Enter marks for painting (0-100)"))

total = math + science + painting
percentage = (total * 100) / 300

print("Total percentage: ", percentage)
print("Total percentage in int: ", int(percentage))
print("Total rounded percentage: ", round(percentage))

grade =None
if percentage >= 80:
    grade = 'A'
elif 80 > percentage >= 70:   # you can do percentage < 80 and percentage >= 70. In Python, you can do chained comparison like this.
    grade = 'B'
elif 70 > percentage >=60:
    grade = 'C'
else:
    grade = 'D'

print("Grade is ", grade)
