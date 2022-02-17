import unittest

"""
name of the test method must start with 'test'
"""
class SimpleTest(unittest.TestCase):
    def testAbc(self):
        self.assertTrue(1 == 1)


if __name__ == '__main__':
    unittest.main()
