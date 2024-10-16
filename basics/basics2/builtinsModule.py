'''

builins.py (builtins module)
----------
    This module is by default available to your module (code in your .py file), just like java.lang package in java.


	In python module means a python file. you can import one python file into another.

	https://docs.python.org/3/library/builtins.html
	All builtin functions and vars are in builins.py. It is by default available to your modules. It is like java.lang.* in Java.
	You dont need to import it in java, so as builtins in Python.


	1) builtins module's classes
	builtins module has classes also.

	2) builtins module's methods
	https://docs.python.org/3/library/functions.html#built-in-funcs

	Many buitins methods are actually classes
	e.g. slice, range, list, __generator, __function, __method, __coroutine, __namedtuple

	Many builtins methods are actually methods also
	e.g. sorted, abs, all, any, callable, compile, setattr, getattr, delattr, dir, eval, exec, exit, format, globals, locals, isinstance, issubclass, iter, len, max, min, open, pow, print, round, __import__, __build_class__, vars etc

	3) builtins module's constants
	https://docs.python.org/3/library/constants.html#built-in-consts

	object class is a base class in Python. It has many methods, but __new__ and __init__ are special.
	__new__ is a method that is used to instantiate a class. Normally, you don't override it in your class.
	__init__(self,...) is a method that is used to initialize instance variables. Normally, you override this method in your class.

	http://stackoverflow.com/questions/674304/pythons-use-of-new-and-init

	__new__ is the first step of instance creation. It's called first, and is responsible for returning a new instance of your class.
	In contrast, __init__ doesn't return anything; it's only responsible for initializing the instance after it's been created.
	In general, you shouldn't need to override __new__ unless you're subclassing an immutable type like str, int, unicode or tuple.



	4) Normally, you don't need to import builtins module in your module, but if you want you can do it to differentiate a call to your function and builtins module's function.
	e.g. this script has open function that needs to call builtins open.
		import builtins

		def open(path):
			f = builtins.open(path)

	You can also use __builitins__.open(path).

	Most modules have the name __builtins__ made available as part of their globals. The value of __builtins__ is normally either this module or the value of this module’s __dict__ attribute.


'''
