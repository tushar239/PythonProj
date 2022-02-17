"""
https://www.geeksforgeeks.org/unit-testing-python-unittest/
https://www.geeksforgeeks.org/python-exceptional-conditions-testing-in-unit-tests/
"""

import unittest
import errno


def parse_int(s):
    return int(s)


class TestAssertRaises(unittest.TestCase):

    def test_assert_raises1(self):
        s = 'hello world'
        with self.assertRaises(TypeError):  # assertRaises returns true, if code in its with block raises TypeError
            s.split(2)  # TypeError: must be str or None, not int
        # is same as
        # self.assertRaises(TypeError, s.split, 2)

    def test_assert_raises2(self):
        # self.assertRaises(ValueError, self.parse_int, "1a")  # '1a' can't be converted to string, parse_int("1a") throws ValueError.
        # same as
        with self.assertRaises(ValueError):
            parse_int("1a")

    # Alternative approach to assertRaises
    def test_alternative_of_assertRaises1(self):
        try:
            f = open('/file/not/found')
        except IOError as e:
            self.assertEqual(e.errno, errno.ENOENT)
        else:
            self.fail('IOError not raised')

    # Alternative approach to assertRaises
    def test_alternative_of_assertRaises2(self):
        try:
            r = parse_int('N/A')
        except ValueError as e:
            self.assertEqual(type(e), ValueError)
        else:
            self.fail('ValueError not raised')

    def test_assertRaisesRegex(self):
        self.assertRaisesRegex(
            ValueError, 'invalid literal .*',
            parse_int, 'N/A')
        # is same as
        """
        with self.assertRaisesRegex(ValueError, 'invalid literal .*'):
            r = parse_int('N/A')
        """


if __name__ == '__main__':
    unittest.main()
