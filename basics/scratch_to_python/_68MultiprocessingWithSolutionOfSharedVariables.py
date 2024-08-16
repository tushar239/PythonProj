"""
Shared memory : multiprocessing module provides Array and Value objects to share data between processes.
    - Array: a ctypes array allocated from shared memory.
    - Value: a ctypes object allocated from shared memory.
Given below is a simple example showing use of Array and Value for sharing data between processes.
"""

import multiprocessing


# function to square a given list
def square_list(mylist, result, square_sum):
    # append squares of mylist to result array
    for idx, num in enumerate(mylist):
        result[idx] = num * num

    # square_sum value
    square_sum.value = sum(result)

    # print result Array
    print("Result(in process p1): {}".format(result[:]))

    # print square_sum Value
    print("Sum of squares(in process p1): {}".format(square_sum.value))


if __name__ == "__main__":
    # input list
    mylist = [1, 2, 3, 4]

    # creating Array of int data type with space for 4 integers
    result = multiprocessing.Array('i', 4)

    # creating Value of int data type
    square_sum = multiprocessing.Value('i')

    # creating new process
    p1 = multiprocessing.Process(target=square_list, args=(mylist, result, square_sum))

    # starting process
    p1.start()

    # wait until the process is finished
    p1.join()

    # print result array
    print("Result(in main program): {}".format(result[:]))

    # print square_sum Value
    print("Sum of squares(in main program): {}".format(square_sum.value))

"""
Let us try to understand the above code line by line:

First of all, we create an Array result like this:
    result = multiprocessing.Array('i', 4)
    First argument is the data type. ‘i’ stands for integer whereas ‘d’ stands for float data type.
    Second argument is the size of array. Here, we create an array of 4 elements.

Similarly, we create a Value square_sum like this:
  square_sum = multiprocessing.Value('i')

Here, we only need to specify data type. The value can be given an initial value(say 10) like this:

  square_sum = multiprocessing.Value('i', 10)
  
IMP:
Now, result and square_sum variables are kept in memory that can be shared by main process and p1 process.

IMP:
Server process : Whenever a python program starts, a server process is also started. From there on, whenever a new process is needed, the parent process connects to the server and requests it to fork a new process.
                 A server process can hold Python objects and allows other processes to manipulate them using proxies.
                 Multiprocessing module provides a Manager class which controls a server process. Hence, managers provide a way to create data that can be shared between different processes.
"""

import multiprocessing


def print_records(records):
    for record in records:
        print("Name: {0}\nScore: {1}\n".format(record[0], record[1]))


def insert_record(record, records):
    records.append(record)
    print("New record added!\n")


if __name__ == '__main__':
    with multiprocessing.Manager() as manager:
        # creating a list in server process memory
        # Similarly, you can create a dictionary as manager.dict method
        records = manager.list([('Sam', 10), ('Adam', 9), ('Kevin', 9)])
        # new record to be inserted in records
        new_record = ('Jeff', 8)

        # creating new processes
        p1 = multiprocessing.Process(target=insert_record, args=(new_record, records))
        p2 = multiprocessing.Process(target=print_records, args=(records,))

        # running process p1 to insert new record
        p1.start()
        p1.join()

        # running process p2 to print records
        p2.start()
        p2.join()
