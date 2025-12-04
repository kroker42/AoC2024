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


###############


class TestDay3(unittest.TestCase):
    data = """987654321111111
811111111111119
234234234234278
818181911112111""".split('\n')
    def test_task1(self):
#        banks = [[int(x) for x in line] for line in self.data]
        banks = self.data
        self.assertEqual(int("98"), elftasks.findHighestJoltage(banks[0]))
        self.assertEqual(int("89"), elftasks.findHighestJoltage(banks[1]))
        self.assertEqual(int("78"), elftasks.findHighestJoltage(banks[2]))
        self.assertEqual(int("92"), elftasks.findHighestJoltage(banks[3]))


    def test_task2(self):
        banks = self.data
        self.assertEqual(987654321111, elftasks.findHighestJoltageChain(banks[0]))
        self.assertEqual(811111111119, elftasks.findHighestJoltageChain(banks[1]))
        self.assertEqual(434234234278, elftasks.findHighestJoltageChain(banks[2]))
        self.assertEqual(888911112111, elftasks.findHighestJoltageChain(banks[3]))

###############


class TestDay4(unittest.TestCase):
    data = """..@@.@@@@.
@@@.@.@.@@
@@@@@.@.@@
@.@@@@..@.
@@.@@@@.@@
.@@@@@@@.@
.@.@.@.@@@
@.@@@.@@@@
.@@@@@@@@.
@.@.@@@.@.""".split('\n')
    def test_task1(self):
        self.assertEqual(13, elftasks.forkliftable(self.data))

    def test_task2(self):
        rolls = elftasks.map_rolls(self.data)

        total_lifted = 0
        lifted = 1
        while lifted:
            lifted = elftasks.forklift(rolls)
            total_lifted += lifted

        self.assertEqual(43, total_lifted)
