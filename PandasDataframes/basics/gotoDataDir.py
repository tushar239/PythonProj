import os

curDir = os.getcwd()
print(curDir)

os.chdir(os.curdir + '\..\data')

curDir = os.getcwd()
print(curDir)