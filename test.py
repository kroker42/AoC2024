import unittest

import functools
import operator

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
        while lifted > 0:
            lifted = elftasks.forklift(rolls)
            total_lifted += lifted

        self.assertEqual(43, total_lifted)


###############


class TestDay5(unittest.TestCase):
    data = """3-7
10-14
16-20
12-18
6-6
3-5
2-2

1
5
8
11
17
32""".split('\n')
    def test_task2(self):
        ranges = elftasks.get_ingredient_ranges(self.data)
        self.assertEqual(16, elftasks.merge_ranges(ranges)) # wrong answer

    def test_merge2(self):
        ranges = elftasks.get_ingredient_ranges(self.data)
        self.assertEqual(17, elftasks.merge_ranges2(ranges))


###############


class TestDay6(unittest.TestCase):
    data = """123 328  51 64 
 45 64  387 23 
  6 98  215 314
*   +   *   + """.split('\n')
    def test_task1(self):
        worksheet = [line.split() for line in self.data]
        numbers = elftasks.str2int(worksheet[:-1])
        ops = self.data[-1].split()

        self.assertEqual([33210, 490, 4243455, 401], elftasks.octo_maths(np.array(numbers).T, ops))

    def test_task2(self):
        worksheet = self.data[:-1]
        ops = self.data[-1].split()

        self.assertEqual([8544, 625, 3253600, 1058], elftasks.r2l_octo_maths(worksheet, ops))



###############


class TestDay7(unittest.TestCase):
    data = """.......S.......
...............
.......^.......
...............
......^.^......
...............
.....^.^.^.....
...............
....^.^...^....
...............
...^.^...^.^...
...............
..^...^.....^..
...............
.^.^.^.^.^...^.
...............""".split('\n')
    def test_task1(self):
        beams, splitters = elftasks.paint_by_numbers(self.data)
        self.assertEqual(21, len(splitters))

    def test_task2(self):
        beams = elftasks.quantum_paint_by_numbers(self.data)
        self.assertEqual(40, sum(beams[len(beams) - 1].values()))


###############


class TestDay8(unittest.TestCase):
    data = """162,817,812
57,618,57
906,360,560
592,479,940
352,342,300
466,668,158
542,29,236
431,825,988
739,650,466
52,470,668
216,146,977
819,987,18
117,168,530
805,96,715
346,949,466
970,615,88
941,993,340
862,61,35
984,92,344
425,690,689""".split('\n')
    def test_task1(self):
        points = [[int(x) for x in line.split(',')] for line in self.data]
        points.sort()
        points = np.array(points)

        distances = elftasks.brute_force_distances(points)
#        print(len(distances))
#        for key in sorted(distances):
#            print(key, distances[key])

        clusters = elftasks.find_clusters(distances, 10)

        cluster_sizes = list(sorted([len(c) for c in clusters], reverse=True))
        self.assertEqual(40, functools.reduce(operator.mul, cluster_sizes[0:3]))