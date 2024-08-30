# In Python, file names are same modules and a directory of these modules is called a package.
# so here 'import_packages_and_modules.Phone' is a package and it has Pots, Isdn, G3, G4 modules.
# Each of these modules can have classes or functions that can be imported and used by client script
# There are two ways to import modules
    # 1. create __init__.py in a package and import all modules there and import just a package in client script
    # 2. import all modules directly in client script

from packagesandmodules.Phone.Pots import pots
from packagesandmodules.Phone.Isdn import isdn
from packagesandmodules.Phone.G3 import g3, anotherG3 # G3, anotherG3 are functions
from packagesandmodules.Phone.G4 import G4 # G4 is a class

# now client script that wants to use import_packages_and_modules.Phone package's modules needs to import a import_packages_and_modules.Phone package
# import import_packages_and_modules.Phone as Phone