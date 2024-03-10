from dataclasses import dataclass

'''
Using Data Class (Python 3.7+): 
In Python 3.7 and above the Data Class can be used to return a class with 
automatically added unique methods. The Data Class module has a decorator and functions for automatically adding 
generated special methods such as __init__() and __repr__() in the user-defined classes.

_repr__() is one of the magic methods that returns a printable representation of an object in Python that can be 
customized or predefined, i.e. we can also create the string representation of the object according to our needs.'''

@dataclass
class Book_list:
    name: str
    perunit_cost: float
    quantity_available: int = 0

    # function to calculate total cost
    def total_cost(self) -> float:
        return self.perunit_cost * self.quantity_available


book = Book_list("Introduction to programming.", 300, 3)
x = book.total_cost()

# print the total cost
# of the book
print(x) # 900

# print book details
print(book) # Book_list(name='Introduction to programming.', perunit_cost=300, quantity_available=3)

# 900
Book_list(name='Python programming.',
          perunit_cost=200,
          quantity_available=3)