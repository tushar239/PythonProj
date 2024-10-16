"""
https://www.geeksforgeeks.org/multiprocessing-python-set-1/
https://www.geeksforgeeks.org/difference-between-multithreading-vs-multiprocessing-in-python/
https://www.geeksforgeeks.org/multiprocessing-python-set-2/

Multiprocessing is same as Parallel processing.

What is multiprocessing?

Multiprocessing refers to the ability of a system to support more than one processor at the same time.
Applications in a multiprocessing system are broken to smaller routines that run independently.
The operating system allocates these threads to the processors improving performance of the system.

What is multiprocessing?

    Multiprocessing refers to the ability of a system to support more than one processor at the same time.
    Applications in a multiprocessing system are broken to smaller routines that run independently.
    The operating system allocates these threads to the processors improving performance of the system.

Here, the CPU can easily executes several tasks at once, with each task using its own processor.

It is just like the chef in last situation being assisted by his assistants.
Now, they can divide the tasks among themselves and chef doesn't need to switch between his tasks.

Multithreading
    is a technique where multiple threads are spawned by a process to do different tasks, at about the same time, just one after the other.
    This gives you the illusion that the threads are running in parallel, but they are actually run in a concurrent manner.
    In Python, the Global Interpreter Lock (GIL) prevents the threads from running simultaneously.
Multiprocessing
    is a technique where parallelism in its truest form is achieved.
    Multiple processes are run across multiple CPU cores, which do not share the resources among them.
    Each process can have many threads running in its own memory space.
    In Python, each process has its own instance of Python interpreter doing the job of executing the instructions.

e.g. Process is like an instance of a program (e.g. opening a word file_
Process can have multiple threads inside it (e.g. writing, correcting spellings, printing etc) that can go in parallel.

Each process takes different memory. Threads can share memory.
Process is heavyweight. Starting a process is slower than starting a thread.
IPC (Inter-process communication) is more complicated.

Threads are good for I/O bound tasks. When one thread is waiting to get the response from I/O operation, that thread wait and let another thread run.
They are not easy to kill. So, be careful with memory intensive race conditions. Race condition can occur when multiple threads want to modify and read the same variable.

Processes have process ids. They are easily killable or interruptable.

Disadvantage of threads in Python:
Python is based on CPython. There is a concept of GIL (Global interpreter lock) in CPython which is needed for memory management.
Unlike to Java, Python threads don't copy shared variables in their caches, so every thread has to change the value of a variable in main memory and this can lead to a big problem with multithreading.
GIL puts automatic lock in the thread when it tries to access shared variable when another thread is accessing it.
So, basically, for some time, threads would not run in parallel.
Watch 'https://www.youtube.com/watch?v=f9q5m321iEU' for more understanding.

To avoid this problem:
- Use multiprocessing
- Use a different, free-threaded Python implementation (Jython, IronPython)
- Use Python as a wrapper for third-party libraries (C/C++) - numpy, scipy
"""

import multiprocessing
import os


def print_cube(num):
    print("ID of process running print_cube: {}".format(os.getpid()))
    # function to print cube of given num
    print("Cube: {}".format(num * num * num))


def print_square(num):
    print("ID of process running print_square: {}".format(os.getpid()))
    # function to print square of given num
    print("Square: {}".format(num * num))


if __name__ == "__main__":
    print("ID of main process: {}".format(os.getpid()))

    # creating processes
    p1 = multiprocessing.Process(target=print_square, args=(10,))
    p2 = multiprocessing.Process(target=print_cube, args=(10,))

    # starting process 1
    p1.start()
    # starting process 2
    p2.start()

    # process IDs
    print("ID of process p1: {}".format(p1.pid))
    print("ID of process p2: {}".format(p2.pid))

    # wait until process 1 is finished
    p1.join()
    # wait until process 2 is finished
    p2.join()

    # both processes finished
    print("Both processes finished execution!")

    # check if processes are alive
    print("Process p1 is alive: {}".format(p1.is_alive()))
    print("Process p2 is alive: {}".format(p2.is_alive()))

    # both processes finished
    print("Done!")

"""
O/P:
ID of main process: 28628
ID of process running worker1: 29305
ID of process running worker2: 29306
ID of process p1: 29305
ID of process p2: 29306
Both processes finished execution!
Process p1 is alive: False
Process p2 is alive: False
"""