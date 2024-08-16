"""
http://www.tutorialspoint.com/python/python_modules.htm


import modulename.classname --- if you use this approach, then you need to use modulename.classname()
from modname import classname1[, classname2[, ... classnameN]] --- if you use this approach, then you can use classname()
from modname import * --- if you use this approach, then you can use classname()

Locating Modules:

When you import a module, the Python interpreter searches for the module in the following sequences −

- The current directory.
- If the module isn't found, Python then searches each directory in the shell variable PYTHONPATH.
- If all else fails, Python checks the default path. On UNIX, this default path is normally /usr/local/lib/python/.

The module search path is stored in the system module sys as the sys.path variable.
The sys.path variable contains the current directory, PYTHONPATH, and the installation-dependent default.

The PYTHONPATH is an environment variable, consisting of a list of directories.
The syntax of PYTHONPATH is the same as that of the shell variable PATH.

Here is a typical PYTHONPATH from a Windows system:

set PYTHONPATH=c:\python20\lib;
And here is a typical PYTHONPATH from a UNIX system:

set PYTHONPATH=/usr/local/lib/python

"""

# import packagesandmodules.library.calculator # importing calculator module from packagesandmodules.library package
# from packagesandmodules.library import *  # importing all modules from packagesandmodules.library package
from packagesandmodules.library import apple, calculator # importing classes
from packagesandmodules.library.pineapple import get_pineapple # importing a function from pineapple module

print(apple.__name__) # packagesandmodules.library.apple

calc = calculator.Calculator()
calc.set_name("Tushar")
print(calc.get_name())

app = apple.Apple()
app.set_color("red")
print(app.get_color())

import packagesandmodules.Phone as Phone # importing a package. To import a package, package needs to have __init__.py with all modules imported in it.

print("calling imported functions: ")
get_pineapple() # I'm a Pineapple
Phone.pots() # I'm Pots Phone
Phone.isdn() # I'm Isdn Phone
Phone.g3() # I'm G3 Phone
g4 = Phone.G4()
g4.get_g4() # I'm G4 Phone

import sys # it is a builtin module for all environment specific variables

# sys is like a environment module and it has environment specific variables
# sys.path.append("/home/me/mypy") # you can add your modules in sys.path
print("system path: " + str(sys.path)) # ['/Users/chokst/MavenizedProjectEclipseWSNew/PythonProject/module/client', '/Users/chokst/MavenizedProjectEclipseWSNew/PythonProject', '/Library/Frameworks/Python.framework/Versions/3.5/lib/python35.zip', '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5', '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/plat-darwin', '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/lib-dynload', '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages']
print("system modules: " + str(sys.modules)) # {'_imp': <module '_imp' (built-in)>, 'os.path': <module 'posixpath' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/posixpath.py'>, 'module.library.apple': <module 'module.library.apple' from '/Users/chokst/MavenizedProjectEclipseWSNew/PythonProject/module/library/apple.py'>, 'sre_constants': <module 'sre_constants' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/sre_constants.py'>, '_bootlocale': <module '_bootlocale' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/_bootlocale.py'>, 'encodings.latin_1': <module 'encodings.latin_1' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/encodings/latin_1.py'>, 'posix': <module 'posix' (built-in)>, 'site': <module 'site' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site.py'>, '_warnings': <module '_warnings' (built-in)>, 'encodings.aliases': <module 'encodings.aliases' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/encodings/aliases.py'>, 'abc': <module 'abc' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/abc.py'>, 'module.library.calculator': <module 'module.library.calculator' from '/Users/chokst/MavenizedProjectEclipseWSNew/PythonProject/module/library/calculator.py'>, '_stat': <module '_stat' (built-in)>, 'sre_compile': <module 'sre_compile' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/sre_compile.py'>, '_collections_abc': <module '_collections_abc' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/_collections_abc.py'>, '_sysconfigdata': <module '_sysconfigdata' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/_sysconfigdata.py'>, '_frozen_importlib': <module '_frozen_importlib' (frozen)>, '_locale': <module '_locale' (built-in)>, '__main__': <module '__main__' from '/Users/chokst/MavenizedProjectEclipseWSNew/PythonProject/module/client/calculatorclient.py'>, 'copyreg': <module 'copyreg' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/copyreg.py'>, 'encodings': <module 'encodings' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/encodings/__init__.py'>, 'sys': <module 'sys' (built-in)>, '_codecs': <module '_codecs' (built-in)>, 'codecs': <module 'codecs' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/codecs.py'>, 'marshal': <module 'marshal' (built-in)>, '_io': <module 'io' (built-in)>, '_frozen_importlib_external': <module '_frozen_importlib_external' (frozen)>, '_weakref': <module '_weakref' (built-in)>, 'module': <module 'module' (namespace)>, 'module.library': <module 'module.library' (namespace)>, '_signal': <module '_signal' (built-in)>, 're': <module 're' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/re.py'>, 'os': <module 'os' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/os.py'>, 'errno': <module 'errno' (built-in)>, 'zipimport': <module 'zipimport' (built-in)>, '_thread': <module '_thread' (built-in)>, 'io': <module 'io' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/io.py'>, 'builtins': <module 'builtins' (built-in)>, 'encodings.utf_8': <module 'encodings.utf_8' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/encodings/utf_8.py'>, '_sitebuiltins': <module '_sitebuiltins' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/_sitebuiltins.py'>, '_sre': <module '_sre' (built-in)>, 'genericpath': <module 'genericpath' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/genericpath.py'>, '_osx_support': <module '_osx_support' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/_osx_support.py'>, 'stat': <module 'stat' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/stat.py'>, 'posixpath': <module 'posixpath' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/posixpath.py'>, '_weakrefset': <module '_weakrefset' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/_weakrefset.py'>, 'sysconfig': <module 'sysconfig' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/sysconfig.py'>, 'sre_parse': <module 'sre_parse' from '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/sre_parse.py'>}
print(dir(sys))
print(dir(Phone))
print(dir(apple))
