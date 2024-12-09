import unittest
import elftasks

###############


class TestDay1(unittest.TestCase):
    def test_task(self):
        lists = [[3,4,2,1,3,3], [4,3,5,3,9,3]]
        self.assertEqual(11, elftasks.sum_min_distances(lists))
        self.assertEqual(31, elftasks.similarity_score(lists))



###############


class TestDay2(unittest.TestCase):
    data = """7 6 4 2 1
1 2 7 8 9
9 7 6 2 1
1 3 2 4 5
8 6 4 4 1
1 3 6 7 9"""
    def test_task(self):
        reports = [list(map(int, line.split())) for line in self.data.split('\n')]
        self.assertEqual([1, 2, 2, 1], elftasks.diff_levels(reports[0]))
        self.assertEqual([1, 0, 0, 0, 0, 1], [elftasks.is_safe(r) for r in reports])
        self.assertEqual([1, 0, 0, 1, 1, 1], [elftasks.dampen_report(r) for r in reports])


###############


class TestDay3(unittest.TestCase):
    def test_task1(self):
        self.assertEqual([('2', '34')], elftasks.multiply_re("mul(2,34)"))
        instr = "xmul(2,4)%&mul[3,7]!@^do_not_mul(5,5)+mul(32,64]then(mul(11,8)mul(8,5))"
        self.assertEqual([('2', '4'), ('5', '5'), ('11', '8'), ('8', '5')], elftasks.multiply_re(instr))
        self.assertEqual([8, 25, 88, 40], elftasks.multiply(elftasks.multiply_re(instr)))

    def test_do_re(self):
        instr = "xmul(2,4)&mul[3,7]!^do()mul(1,2)don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))do()mul(3,4)don't()mul(5,6)"
        self.assertEqual(['mul(2,4)',
 'do()',
 'mul(1,2)',
 "don't()",
 'mul(5,5)',
 'mul(11,8)',
 'do()',
 'mul(8,5)',
 'do()',
 'mul(3,4)',
 "don't()",
 'mul(5,6)'], elftasks.do_re(instr))


###############


class TestDay4(unittest.TestCase):
    data = """MMMSXXMASM
MSAMXMSMSA
AMXSXMAAMM
MSAMASMSMX
XMASAMXAMM
XXAMMXXAMA
SMSMSASXSS
SAXAMASAAA
MAMMMXMMMM
MXMXAXMASX""".split('\n')

    def test_grid(self):
        grid = elftasks.Grid(self.data)
        self.assertEqual('M', grid.get(0, 0))
        self.assertEqual('', grid.get(0, -1))
        self.assertEqual('', grid.get(1, 10))
        self.assertEqual('S', grid.get(2, 3))

    def test_count_xmas(self):
        grid = elftasks.Grid(self.data)
        self.assertEqual(1, elftasks.count_xmas(grid, 0, 5))

    def test_task1(self):
        grid = elftasks.Grid(self.data)
        self.assertEqual(18, grid.count(elftasks.count_xmas))

    def test_task2(self):
        grid = elftasks.Grid(self.data)
        self.assertEqual(9, grid.count(elftasks.count_x_mas))

###############


class TestDay5(unittest.TestCase):
    rules = """47|53
97|13
97|61
97|47
75|29
61|13
75|53
29|13
97|29
53|29
61|53
97|53
61|29
47|13
75|47
97|75
47|61
75|61
47|29
75|13
53|13

75,47,61,53,29
97,61,53,29,13
75,29,13
75,97,47,61,53
61,13,29
97,13,75,29,47"""
    def test_task1(self):
        rulebook = elftasks.PrintRules([line.strip() for line in self.rules.split('\n')])
        self.assertEqual((0, 0), rulebook.validate_print_run([75,47,61,53,29]))
        self.assertEqual((1, 4), rulebook.validate_print_run([97,13,75,29,47]))

###############


class TestDay6(unittest.TestCase):
    data = """....#.....
.........#
..........
..#.......
.......#..
..........
.#..^.....
........#.
#.........
......#..."""
    def test_task1(self):
        grid = self.data.split('\n')
        self.assertEqual((6, 4), elftasks.get_map(grid))

        obstacle, path = elftasks.find_obstacle((6, 4), (-1, 0), grid)
        self.assertEqual([0, 4], obstacle)
        self.assertEqual({(i,4) for i in range(1, 7)}, path)

        obstacle, path = elftasks.find_obstacle((7, 7), (1, 0), grid)
        self.assertEqual([10, 7], obstacle)

        self.assertEqual(41, len(elftasks.path_length((6, 4), grid)))



###############


class TestDay7(unittest.TestCase):
    data = """190: 10 19
3267: 81 40 27
83: 17 5
156: 15 6
7290: 6 8 6 15
161011: 16 10 13
192: 17 8 14
21037: 9 7 18 13
292: 11 6 16 20"""


    def test_task1(self):
        equations = elftasks.get_equations(self.data.split('\n'))

###############


class TestDay8(unittest.TestCase):
    data = """..........
..........
..........
....a.....
..........
.....a....
..........
..........
..........
.........."""
    grid = data.split('\n')

    def test_task1(self):
        masts = elftasks.find_masts(self.grid)
        self.assertEqual({'a': [(3, 4), (5, 5)]}, masts)
        self.assertEqual({(1, 3), (7, 6)}, elftasks.find_antinodes(masts))

    def test_antinode_outside_grid(self):
        masts = elftasks.find_masts(self.grid)
        masts['a'].append((4, 8))
        self.assertEqual({(1, 3), (7, 6), (6, 2), (3, 11), (2, 0), (5, 12)}, elftasks.find_antinodes(masts))

    def test_antinode_different_mast_types(self):
        grid = """............
........0...
.....0......
.......0....
....0.......
......A.....
............
............
........A...
.........A..
............
............""".split('\n')
        masts = elftasks.find_masts(grid)
        antinodes = elftasks.nodes_in_grid(set(elftasks.find_antinodes(masts)), grid)
        self.assertEqual({(0, 6), (0, 11), (1, 3), (2, 4), (2, 10), (3, 2), (4, 9), (5, 1), (5, 6), (6, 3), (7, 0), (7, 7), (10, 10), (11, 10)},
                         set(antinodes))


###############


class TestDay9(unittest.TestCase):
    def test_task1(self):
        disk = [int(x) for x in "2333133121414131402"]
        compressed_disk = elftasks.fragment_disk(disk)
        self.assertEqual([int(x) for x in '0099811188827773336446555566'], compressed_disk)
        self.assertEqual(1928, elftasks.checksum(compressed_disk))

    def test_task2(self):
        self.assertEqual(2858, elftasks.checksum(elftasks.compress_disk([int(x) for x in "2333133121414131402"])))
        self.assertEqual(2858, elftasks.checksum([int(x) for x in "0099211177704403330000555506666000008888"]))

