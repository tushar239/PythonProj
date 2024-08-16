a = input("Enter number: ")

if a.isdigit() and int(a) > 0:
    print(a, " is a positive number")
elif a.isdigit() and int(a) < 0:
    print(a, " is a negative number")
elif a.isdigit() and int(a) == 0:
    print(a, " is zero")
else:
    print(a, " is not a number")

