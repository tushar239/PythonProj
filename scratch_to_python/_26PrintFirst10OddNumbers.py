# you can write this program without using list also, but for practice, we have used list

oddnumbers = list()

i = 1
while len(oddnumbers) <= 10:
    if not i % 2 == 0:  # i % 2 != 0 would also work
        oddnumbers.append(i)
    i = i + 1

print(oddnumbers)
