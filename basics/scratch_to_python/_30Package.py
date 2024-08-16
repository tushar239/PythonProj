"""
Concept of package in Python is same as Java.
In Java, when you want external library, you need a jar file. Python doesn't have a concept like jar file. You just install a package related to that library from github.
Certain packages are available in Python implicitly, others you have to install.
e.g. if you want to use numpy, PyTime etc packages
you have to first find this package in github
command to install a package - C:\Projects\PythonProject\venv\Scripts>pip install numpy

In PyCharm, it is a lot easier
go to settings -> your project -> Python interpreter -> click '+' sign to install a package -> search 'numpy' and install.

When you install a particular package, all related packages on which that package is dependent, will also be installed.
"""

""" 
you can import entire package(library) or a class or a function
import numpy ---- this is like 'import numpy.*' in jaa
from numpy.core import ComplexWarning ---- this is like 'import numpy.core.ComplexWarning' in jaa
from numpy.core.numeric import full --- you can't import just a function in java
"""

