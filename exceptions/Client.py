"""
A single try statement can have multiple except statements.
else block:
    After the except clause(s), you can include an else-clause.
    The code in the else-block executes if the code in the try block does not raise an exception.
    The else-block is a good place for code that does not need the try: block's protection.
finally block:
    It is just like Java's finally block.


try:
   You do your operations here;
   ......................
except ExceptionI:
   If there is ExceptionI, then execute this block.
except ExceptionII:
   If there is ExceptionII, then execute this block.
   ......................
else:
   If there is no exception then execute this block.
finally:
    This would always be executed.


Handling multiple exceptions together

try:
   You do your operations here;
   ......................
except(Exception1[, Exception2[,...ExceptionN]]]):
   If there is any exception from the given exception list,
   then execute this block.
   ......................
else:
   If there is no exception then execute this block.
finally:
    This would always be executed.

try-finally (without except)

try:
   You do your operations here;
finally:
    This would always be executed.

Raising an Exceptions:
    raise [Exception [, args [, traceback]]]

    An exception can be a string, a class or an object.



"""

import random
import os

try:
    # open a file only for reading

    randomNo = random.randint(1, 10)
    fileName = "onlyforread" + str(randomNo) + ".txt"
    if(os._exists()):
        os.remove(fileName) # just want to raise an exception and trying to open a file after deleting it
    file = open(fileName, "r")
except IOError:
    print("file doesn't exist")
except:
    print("some other exception")
#else:
    #  else block is executed, if exception is not raised from try block
finally:
    file = open("onlyforread" + str(randomNo) + ".txt", "w+")
    file.write("I am happy")

    if(not file.closed):
        file.close()


# Example of raising an error using custom error class
# Custom Error class extending RuntimeError
class Networkerror(RuntimeError):
    def __init__(self, arg):
        assert(arg  != None), "No error message supplied"
        super().__init__(arg)

try:
   raise Networkerror("Bad hostname")
except Networkerror as e:
   # print(e.__str__())
   # or
   print(e)


# Example of raising a string error
def fun(level):
    try:
       if(not level or level == 0):
           raise "Invalid level!"

    except "Invalid level!" as e:
       print(e)
    else:
       print("level: "+str(level))

#fun(0) # to raise an exception
fun(1)