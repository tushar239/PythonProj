'''
https://www.kodeclik.com/python-file-naming-convention/

Naming Python modules

A Python module is a collection of reusable code that is inside a file with extension “.py” and which can be imported inside another Python program.
Typically, a module contains a set of functions that is helpful to break down a program into manageable parts.
Similar to files and functions, modules should be named in lowercase characters with an underscore used when the name is too long.
It is recommended that modules be named with just a single word.


Naming Python packages

A Python package is a collection of modules.
It is typically a folder containing modules and maybe other folders each of which can contain more modules and/or folders.
A package folder typically contains a file called “__init__.py” which is a way to signal to Python that this folder is a package.
The contents of “__init__.py” typically contain code to be executed upon package initialization within a larger program.
'''

'''
In Python, the __init__.py file is used to mark a directory as a Python package. 
It is used to initialize the package when it is imported. The __init__.py file can contain code that will be executed when the package is 
imported, as well as function definitions and variable assignments.
'''

'''
https://stackoverflow.com/questions/448271/what-is-init-py-for

Files named __init__.py are used to mark directories on disk as Python package directories. 
If you a package span like below in your project

    spam/__init__.py
    spam/module.py

you can import the code in module.py as
    
    import spam.module
    or
    from spam import module

If you remove the __init__.py file, Python will no longer look for submodules inside that directory, 
so attempts to import the module will fail.

The __init__.py file is usually empty, but can be used to export selected portions of the package under 
more convenient name, hold convenience functions, etc. 
Given the example above, the contents of the init module can be accessed as

    import spam
    

There are 2 main reasons for __init__.py
1. For convenience: the other users will not need to know your functions' exact location in your package hierarchy
    your_package/
          __init__.py
          file1.py
          file2.py
            ...
          fileN.py
    
    # in __init__.py
        from .file1 import *
        from .file2 import *
        ...
        from .fileN import *
    
    # in file1.py
        def add():
            pass

    then others can call add() by
    
        from your_package import add
    
    without knowing file1's inside functions, like
    
        from your_package.file1 import add

'''