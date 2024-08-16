# you can write this program without using list also, but for practice, we have used list

evennumbers = list()

i = 1
while len(evennumbers) <= 10:
    if i % 2 == 0:
        evennumbers.append(i)
    i = i + 1

print(evennumbers)
