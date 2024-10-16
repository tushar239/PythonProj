def getPairsCount(arr, sum):
    count = 0  # Initialize result

    # Consider all possible pairs
    # and check their sums
    for i in range(0, len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] + arr[j] == sum:
                count += 1

    return count

# Driver function
arr = [1, 5, 7, -1, 5]
sum = 6
print("Count of pairs is",
      getPairsCount(arr, sum))
