"""
https://www.geeksforgeeks.org/what-does-the-if-__name__-__main__-do/

Before executing code, Python interpreter reads source file and define few special variables/global variables.
If the python interpreter is running that module (the source file) as the main program, it sets the special __name__ variable to have a value “__main__”.
If this file is being imported from another module, __name__ will be set to the module’s name. Module’s name is available as value to __name__ global variable.

A module is a file containing Python definitions and statements. The file name is the module name with the suffix .py appended.
"""

print("Always executed")

if __name__ == "__main__":
    print("Executed when invoked directly")
else:
    print("Executed when imported, __name__ is set as", __name__)  # Executed when imported, __name__ is set as _76WhatIs__main__
