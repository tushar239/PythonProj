# https://www.geeksforgeeks.org/__new__-in-python/

class A(object):

    def __new__(cls):
        print("Creating instance")

    # It is not called. Internally __new__() calls __init__(), but here you have overridden __new__()
    def __init__(self):
        print("Init is called")


print(A())

'''
Creating instance
None
'''