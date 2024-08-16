"""
https://www.geeksforgeeks.org/multithreading-python-set-1/

This article covers the basics of multithreading in Python programming language. Just like multiprocessing, multithreading is a way of achieving multitasking. In multithreading, the concept of threads is used.

In simple words, a thread is a sequence of such instructions within a program that can be executed independently of other code.
For simplicity, you can assume that a thread is simply a subset of a process!

Multithreading is defined as the ability of a processor to execute multiple threads concurrently.

In a simple, single-core CPU, it is achieved using frequent switching between threads. This is termed as context switching.
Context switching takes place so frequently that all the threads appear to be running parallelly (this is termed as multitasking).

IMP:
Unlike to Java, Python Threads do not carry their own caches. So, there is no need of 'volatile' variable in Python.
Java Threads concurrency issue: https://www.baeldung.com/java-volatile

"""

# Python program to illustrate the concept
# of threading
# importing the threading module
import threading

import time


def print_cube(num):
    print("Inside print_cube()")
    print("Cube: {}".format(num * num * num))


def print_square(num):
    print("Inside print_square()")
    #for _ in range(10000000000000):
        #pass
    # time.sleep(2)  # adding wait period of 2 secs
    print("Square: {}".format(num * num))


if __name__ == "__main__":
    # creating thread
    t1 = threading.Thread(target=print_square, args=(10,))
    t2 = threading.Thread(target=print_cube, args=(10,))

    # starting thread 1
    t1.start()
    # starting thread 2
    t2.start()

    # this code is written inside main thread, so main thread will wait until thread 1 is completely executed
    t1.join()
    # this code is written inside main thread, so main thread will wait until thread 2 is completely executed
    t2.join()

    # both threads completely executed
    print("Done!")


