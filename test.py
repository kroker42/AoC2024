import unittest
import elftasks
import numpy as np



###############


class TestDay1(unittest.TestCase):
    data = """""".split('\n')
    def test_task1(self):
        self.assertEqual(True, True)

###############


class TestDay2(unittest.TestCase):
    def test_task1(self):
        self.assertEqual([22, 33, 44], elftasks.find_invalid_ids('12', '45'))
        self.assertEqual(9, len(elftasks.find_invalid_ids('1', '110')))

    def test_task2(self):
        self.assertEqual([11, 22], elftasks.find_repetitive_ids('11', '22'))
        self.assertEqual([99, 111], sorted(elftasks.find_repetitive_ids('95', '115')))
        self.assertEqual([999, 1010], sorted(elftasks.find_repetitive_ids('998', '1012')))
        self.assertEqual([1188511885], sorted(elftasks.find_repetitive_ids('1188511880', '1188511890')))
