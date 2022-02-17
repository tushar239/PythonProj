"""
https://www.geeksforgeeks.org/multiprocessing-python-set-2/

In multiprocessing, any newly created process will do following:
    - run independently
    - have their own memory space.
"""

"""
O/P of following program will be:
Result(in process p1): [1, 4, 9, 16]
Result(in main program): []

Since, main process and p1 process use different memory, p1 is changing the list, but tha change will never get reflected in main process' memory.
So, result will still be empty in main process.
"""

import multiprocessing

# empty list with global scope
result = []


def square_list(mylist):
    global result
    # append squares of mylist to global list result
    for num in mylist:
        result.append(num * num)
    # print global list result
    print("Result(in process p1): {}".format(result))


if __name__ == "__main__":
    # input list
    mylist = [1, 2, 3, 4]

    # creating new process
    p1 = multiprocessing.Process(target=square_list, args=(mylist,))
    # starting process
    p1.start()
    # wait until process is finished
    p1.join()

    # print global result list
    print("Result(in main program): {}".format(result))
