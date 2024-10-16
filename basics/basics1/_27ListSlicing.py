"""
List - https://www.youtube.com/watch?v=fAr6EMp0SSc
"""

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

print(letters[0])  # A
print(letters[:7])  # ['A', 'B', 'C', 'D', 'E', 'F', 'G'] --- printing 0 to 6 elements
print(letters[1:7])  # ['B', 'C', 'D', 'E', 'F', 'G'] --- printing 1 to 6 elements
print(letters[-5:-1])  # ['F', 'G', 'H', 'I'] --- printing -5 to -2 elements
print(letters[1:7:2])  # ['B', 'D', 'F'] --- 3rd parameter is step. printing 1st, 3rd, 5th elements
print(letters[::-1])  # ['J', 'I', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']   --- printing end to start (reversing a list)

print(letters[:0])  # this is same as letters[0:0]
