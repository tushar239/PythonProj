import unittest


class TestStringMethods(unittest.TestCase):

    def setUp(self):
        print("setUp() is called")

    # Returns True if the string contains 4 a.
    def test_strings_a(self):
        self.assertEqual('a' * 4, 'aaaa')

    # Returns True if the string is in upper case.
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    # Returns TRUE if the string is in uppercase
    # else returns False.
    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    # Returns true if the string is stripped and
    # matches the given output.
    def test_strip(self):
        s = 'geeksforgeeks'
        self.assertEqual(s.strip('geek'), 'sforgeeks')

    # Returns true if the string splits and matches the given output.
    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        with self.assertRaises(TypeError):  # assertRaises returns true, if code in its with block raises TypeError
            s.split(2)   # TypeError: must be str or None, not int
        # is same as
        # self.assertRaises(TypeError, s.split, 2)

    def parse_int(self, s):
        return int(s)

    def test_bad_int(self):
        # self.assertRaises(ValueError, self.parse_int, "1a")  # '1a' can't be converted to string, parse_int("1a") throws ValueError.
        # same as
        with self.assertRaises(ValueError):
            self.parse_int("1a")

    # https://www.geeksforgeeks.org/python-unittest-assertin-function/
    def test_assertIn(self):
        key = "gfg"
        container = "geeksforgeeks"
        # error message in case if test case got failed
        message = "key is not in container."
        # assertIn() to check if key is in container
        self.assertIn(key, container, message)  # AssertionError: 'gfg' not found in 'geeksforgeeks' : key is not in container.
        self.assertNotIn(key, container, message)

    def test_assertNotIn(self):
        key = "gfg"
        container = "geeksforgeeks"
        # error message in case if test case got failed
        message = "key is in container."
        # assertNotIn() to check if key is not in the container
        self.assertNotIn(key, container, message)

if __name__ == '__main__':
    unittest.main()
