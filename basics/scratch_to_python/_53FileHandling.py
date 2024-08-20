"""
The key function for working with files in Python is the open() function.

The open() function takes two parameters; filename, and mode.

There are four different methods (modes) for opening a file:

    "r" - Read - Default value. Opens a file for reading, error if the file does not exist
    "a" - Append - Opens a file for appending, creates the file if it does not exist
    "w" - Write - Opens a file for writing, creates the file if it does not exist. It will overwrite the entire file.
    "x" - Create - Creates the specified file, returns an error if the file exists. It creates an empty file

In addition, you can specify if the file should be handled as binary or text mode

    "t" - Text - Default value. Text mode
    "b" - Binary - Binary mode (e.g. images)
"""
import os

filelocation = "/temp/demofile.txt"

os.makedirs("/temp")
f = open(filelocation, "w")  # default value is "rt". You can't have 'rw'. More than one mode can't be specified.
f.write("Now the file has more content!")
f.close()

f = open(filelocation, "r")
print(f.read())
f.close()


if os.path.exists(filelocation):
    os.remove(filelocation)
    os.rmdir("/temp")
else:
    print("The file does not exist")
