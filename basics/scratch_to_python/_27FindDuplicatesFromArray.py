def findDuplicates(thislist):
    for i in range(0, len(thislist)):
        element = thislist[i]
        for j in range(i + 1, len(thislist)):
            if element == thislist[j]:
                print(thislist[i])


if __name__ == '__main__':
    thislist: list = [1, 2, 5, 3, 1, 5]
    findDuplicates(thislist);
