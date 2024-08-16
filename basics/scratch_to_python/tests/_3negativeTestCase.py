# unit test case
import unittest


class TestNegativeTest(unittest.TestCase):

    def test_negative(self):
        firstValue = "geeks"
        secondValue = "gfg"
        # error message in case if test case got failed
        message = "First value and second value are not equal !"
        # assertEqual() to check equality of first & second value
        self.assertEqual(firstValue, secondValue, message)  # if test fails, it prints a message


if __name__ == '__main__':
    unittest.main()
