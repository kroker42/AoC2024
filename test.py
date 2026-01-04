import unittest

import itertools
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

    def test_pairwise_distances(self):
        points = [tuple([int(x) for x in line.split(',')]) for line in self.data]
        distances = elftasks.all_pairwise_distances(points)
        clusters = elftasks.find_shortest_clusters(distances)
        self.assertEqual([5, 4, 2, 2], sorted([len(c) for c in clusters]))

    def test_single_cluster(self):
        points = [tuple([int(x) for x in line.split(',')]) for line in self.data]
        distances = elftasks.all_pairwise_distances(points)
        self.assertEqual(25272, elftasks.find_single_cluster(distances, len(points)))


###############


class TestDay9(unittest.TestCase):
    data = """7,1
11,1
11,7
9,7
9,5
2,5
2,3
7,3""".split('\n')
    tiles = [tuple([int(x) for x in tile.split(',')]) for tile in data]

    def test_task1(self):
        self.assertEqual(50, max(elftasks.squarea(self.tiles)))

    def test_task2(self):
        pairs = itertools.combinations(self.tiles, 2)


###############


class TestDay10(unittest.TestCase):
    data = """[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}
[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}""".split('\n')
    def test_task1(self):
        machines = elftasks.parse_factory_machines([line.split(' ') for line in self.data])
        self.assertEqual(['.##.', [[3], [1,3], [2], [2,3], [0,2], [0,1]], '{3,5,4,7}'], machines[0])

###############


class TestDay11(unittest.TestCase):
    data = """aaa: you hhh
you: bbb ccc
bbb: ddd eee
ccc: ddd eee fff
ddd: ggg
eee: out
fff: out
ggg: out
hhh: ccc fff iii
iii: out""".split('\n')
    def test_task1(self):
        devices = [line.split(' ') for line in self.data]
        graph = elftasks.build_device_graph(devices)
        self.assertEqual(["bbb", "ccc"], graph["you"])
        self.assertEqual(["out"], graph["fff"])

        paths = elftasks.Paths(graph)
        self.assertEqual(5, paths.count_paths("you"))

    def test_task2(self):
        data = """svr: aaa bbb
aaa: fft
fft: ccc
bbb: tty
tty: ccc
ccc: ddd eee
ddd: hub
hub: fff
eee: dac
dac: fff
fff: ggg hhh
ggg: out
hhh: out""".split('\n')
        devices = [line.split(' ') for line in data]
        graph = elftasks.build_device_graph(devices)

        paths = elftasks.Paths(graph)
        self.assertEqual(2, paths.count_dac_fft_paths("svr"))




###############


class TestDay12(unittest.TestCase):
    gifts = """0:
###
##.
##.

1:
###
##.
.##

2:
.##
###
##.

3:
##.
###
##.

4:
###
#..
###

5:
###
.#.
###""".split('\n')
    trees = """4x4: 0 0 0 0 2 0
12x5: 1 0 1 0 2 2
12x5: 1 0 1 0 3 2""".split('\n')

    def test_task1(self):
        trees = [tree.split(':') for tree in self.trees]
        tree_spaces = [[int(d) for d in tree[0].split('x')] for tree in trees]
        gifts = [[int(gift) for gift in tree[-1].strip().split(' ')] for tree in trees]

        self.assertEqual(3, sum([elftasks.enough_spaces(sum(gifts[i]), tree_spaces[i]) for i in range(len(trees))]))