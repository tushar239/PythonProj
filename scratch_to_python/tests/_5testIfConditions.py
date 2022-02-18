import unittest
import _5ifconditions as obj

"""
check following link for the use of isinstance() method
https://www.guru99.com/type-isinstance-python.html
"""

class TestIfConditions(unittest.TestCase):
    def testInputValueAsStringDigit(self):
        self.assertEqual(5, obj.fun("5"))

    def testInputValueAsInt(self):
        self.assertEqual(5, obj.fun(5))

    def testInputValueAsWord(self):
        self.assertRaises(TypeError, obj.fun, "hi")

    def testInputValueAsFloat(self):
        self.assertRaises(TypeError, obj.fun, "5.2")

if __name__ == "__main__":
    unittest.main()