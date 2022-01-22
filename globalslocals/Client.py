firstName = None
lastName = None

def set_name(fName, lName="Chokshi"):
    global firstName
    global lastName

    firstName=fName
    lastName=lName
    print(locals()) # {'lastName': 'Chokshi', 'firstName': 'Tushar', 'fName': 'Tushar', 'lName': 'Chokshi'}
    print(globals()) # {'lastName': None, 'set_name': <function set_name at 0x10207b8c8>, '__package__': None, '__builtins__': <module 'builtins' (built-in)>, '__spec__': None, '__name__': '__main__', 'firstName': None, '__doc__': None, '__cached__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x1005a5cf8>, 'get_name': <function get_name at 0x10207b950>, '__file__': '/Users/chokst/MavenizedProjectEclipseWSNew/PythonProject/globalslocals/globalslocals.py'}

def get_name():
    return firstName+" "+lastName

set_name("Tushar")
print(get_name()) #Tushar Chokshi
