'''
https://www.geeksforgeeks.org/find-the-missing-number/
'''

def getMissingNo(arr):
    for i in range(1, len(arr)):
        if arr[i - 1] != arr[i] - 1:
            return arr[i] - 1;
    return None;


def getMissingNo2(arr: list):
    n: int = len(arr)
    total: int = (n + 1) * (n + 2) / 2
    sum_of_arr = sum(arr)
    return total - sum_of_arr


if __name__ == '__main__':
    thislist: list = [1, 2, 3, 5]
    print(getMissingNo(thislist));
    print(getMissingNo2(thislist));