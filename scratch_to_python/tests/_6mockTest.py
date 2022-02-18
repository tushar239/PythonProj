import unittest  # unittest is a packagename
from unittest import mock, main  # unittest package has module 'mock' and variable 'main'. this variable is assigned a class TestProgram. so, when you do main(), it instantiates TestProgram class.
from unittest.mock import MagicMock


class TestMocking(unittest.TestCase):

    def testGetName(self):
        mocked_b = MagicMock()
        mocked_b.getname.return_value = "John"
        a = A(mocked_b)

        self.assertEqual("John, how are you?", a.getname())
        print(mocked_b.getname.assert_called)
        mocked_b.getname.assert_called()
        self.assertEqual(1, mocked_b.getname.call_count)
        # if mocked method takes a parameter, you can also test
        # mocked_b.getname.assert_called_with(parameter value)

if __name__ == "__main__":
    unittest.main()


class A:
    def __init__(self, b):
        self.b = b

    def getname(self):
        return self.b.getname() + ", how are you?"


class B:
    def __init__(self, name):
        self.name = name

    def getname(self):
        return self.name
