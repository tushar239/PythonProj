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

