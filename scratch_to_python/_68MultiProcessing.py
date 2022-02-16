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
"""

import multiprocessing


def print_cube(num):
    """
    function to print cube of given num
    """
    print("Cube: {}".format(num * num * num))


def print_square(num):
    """
    function to print square of given num
    """
    print("Square: {}".format(num * num))


if __name__ == "__main__":
    # creating processes
    p1 = multiprocessing.Process(target=print_square, args=(10,))
    p2 = multiprocessing.Process(target=print_cube, args=(10,))

    # starting process 1
    p1.start()
    # starting process 2
    p2.start()

    # wait until process 1 is finished
    p1.join()
    # wait until process 2 is finished
    p2.join()

    # both processes finished
    print("Done!")