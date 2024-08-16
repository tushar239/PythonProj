"""
https://www.geeksforgeeks.org/multithreading-in-python-set-2-synchronization/

A race condition occurs when two or more threads can access shared data and they try to change it at the same time.
As a result, the values of variables may be unpredictable and vary depending on the timings of context switches of the processes.

import threading

# global variable x
x = 0

def increment():
	# function to increment global variable x
	global x
	x += 1

def thread_task():
	# task for thread calls increment function 100000 times.
	for _ in range(100000):
		increment()

def main_task():
	global x
	# setting global variable x as 0
	x = 0

	# creating threads
	t1 = threading.Thread(target=thread_task)
	t2 = threading.Thread(target=thread_task)

	# start threads
	t1.start()
	t2.start()

	# wait until threads finish their job
	t1.join()
	t2.join()

if __name__ == "__main__":
	for i in range(10):
		main_task()
		print("Iteration {0}: x = {1}".format(i,x))

Output:

Iteration 0: x = 175005
Iteration 1: x = 200000
Iteration 2: x = 200000
Iteration 3: x = 169432
Iteration 4: x = 153316
Iteration 5: x = 200000
Iteration 6: x = 167322
Iteration 7: x = 200000
Iteration 8: x = 169917
Iteration 9: x = 153589

In above program:

Two threads t1 and t2 are created in main_task function and global variable x is set to 0.
Each thread has a target function thread_task in which increment function is called 100000 times.
increment function will increment the global variable x by 1 in each call.
The expected final value of x is 200000 but what we get in 10 iterations of main_task function is some different values.

This happens due to concurrent access of threads to the shared variable x. This unpredictability in value of x is nothing but race condition.

Threading module provides a Lock class to deal with the race conditions.
Lock is implemented using a Semaphore object provided by the Operating System.
"""

import threading

# global variable x
x = 0


def increment():
    # function to increment global variable x
    global x
    x += 1


def thread_task(lock):
    # task for thread calls increment function 100000 times.
    for _ in range(100000):
        lock.acquire()
        increment()
        lock.release()


def main_task():
    global x
    # setting global variable x as 0
    x = 0

    # creating a lock
    lock = threading.Lock()

    # creating threads
    t1 = threading.Thread(target=thread_task, args=(lock,))
    t2 = threading.Thread(target=thread_task, args=(lock,))

    # start threads
    t1.start()
    t2.start()

    # wait until threads finish their job
    t1.join()
    t2.join()


if __name__ == "__main__":
    for i in range(10):
        main_task()
        print("Iteration {0}: x = {1}".format(i, x))

"""
O/P:
Iteration 0: x = 200000
Iteration 1: x = 200000
Iteration 2: x = 200000
Iteration 3: x = 200000
Iteration 4: x = 200000
Iteration 5: x = 200000
Iteration 6: x = 200000
Iteration 7: x = 200000
Iteration 8: x = 200000
Iteration 9: x = 200000
"""