# http://www.tutorialspoint.com/python/python_command_line_arguments.htm

import sys
import builtins

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

def func(required_arg, *args, **kwargs):
    # required_arg is a positional-only parameter.
    print(required_arg)

    # args is a tuple of positional arguments,
    # because the parameter name has * prepended.
    if args: # If args is not empty.
        print(args)

    # kwargs is a dictionary of keyword arguments,
    # because the parameter name has ** prepended.
    if kwargs: # If kwargs is not empty.
        print(kwargs)

func("required argument")
# O/P: required argument
func("required argument", 1, 2, '3')
# O/P:
# required argument
# (1, 2, '3')
func("required argument", 1, 2, '3', hi=4, hello="foo")
# O/P:
# required argument
# (1, 2, '3')
# {'hi': 'foo', 'hello': 4}
print(sum((1,2,3))) # 6
print(sum({4:41,5:51})) # 9

print(__builtins__)  # <module 'builtins' (built-in)>

