import os

curDir = os.getcwd()
print(curDir)

parent_dir = os.path.split(curDir)[0]
print(parent_dir)

os.chdir(parent_dir + '\data')

curDir = os.getcwd()
print(curDir)