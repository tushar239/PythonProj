# you can write this program without using list also, but for practice, we have used list

evennumbers = list()

for i in range(1, 11):
    if i % 2 == 0:
        evennumbers.append(i)

print(evennumbers)
