import numpy as np

a = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])
print("before transpose \n", a)

b = np.transpose(a)

print("after transpose \n", b)
'''
 [[1 4 7]
 [2 5 8]
 [3 6 9]]
'''
# changes the original array 'a' also
b[1][0] = 22

print(b)
print(a)

'''
[[ 1  4  7]
 [22  5  8]
 [ 3  6  9]]
 
[[ 1 22  3]
 [ 4  5  6]
 [ 7  8  9]]
'''