import unittest
import elftasks
import numpy as np

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



###############


class TestDay10(unittest.TestCase):
    def test_task1(self):
        data = [[int(x) for x in line] for line in """89010123
78121874
87430965
96549874
45678903
32019012
01329801
10456732""".split('\n')]
        self.assertEqual(36, elftasks.IslandMap(data).score_trails())

    def test_trivial_data(self):
        data = [[int(x) for x in line] for line in """0123
1234
8765
9876""".split('\n')]
        self.assertEqual(1, elftasks.IslandMap(data).score_trails())


###############


class TestDay11(unittest.TestCase):
    def test_task1(self):
        stones = [int(x) for x in "0 1 10 99 999".split(' ')]
        self.assertEqual([1, 2024, 1, 0, 9, 9, 2021976], elftasks.blink(stones))
        self.assertEqual(len([1, 2024, 1, 0, 9, 9, 2021976]), sum(elftasks.blink_at_buckets(elftasks.count_stones(stones)).values()))

        stones = [int(x) for x in "125 17".split(' ')]
        for i in range(6):
            stones = elftasks.blink(stones)
        exp = [int(x) for x in "2097446912 14168 4048 2 0 2 4 40 48 2024 40 48 80 96 2 8 6 7 6 0 3 2".split(' ')]
        self.assertEqual(exp, stones)

        stones = [int(x) for x in "125 17".split(' ')]
        buckets = elftasks.count_stones(stones)
        for i in range(6):
            buckets = elftasks.blink_at_buckets(buckets)
        self.assertEqual(len(exp), sum(buckets.values()))



###############

class TestDay12(unittest.TestCase):
    data = [ list(x) for x in """RRRRIICCFF
RRRRIICCCF
VVRRRCCFFF
VVRCCCJFFF
VVVVCJJCFE
VVIVCCJJEE
VVIIICJJEE
MIIIIIJJEE
MIIISIJEEE
MMMISSJEEE""".split('\n')]
    def test_task1(self):
        patch = elftasks.GardenPatch(self.data)
        patch.find_regions()

        prices = []
        for region in patch.regions:
            fence = 0
            for coords in region:
                fence += 4 - len(patch.fenced_off[coords])
            prices.append(len(region) * fence)

        self.assertEqual(1930, sum(prices))

    def test_find_sides(self):
        patch = elftasks.GardenPatch(self.data)
        patch.find_regions()

        prices = []

        for region in patch.regions:
            sides = patch.find_sides(region)
            prices.append(len(region) * sides)

        self.assertEqual(1206, sum(prices))

###############


class TestDay13(unittest.TestCase):
    data = """Button A: X+94, Y+34
Button B: X+22, Y+67
Prize: X=8400, Y=5400""".split('\n')
    eq = """8400 = 94 x + 22 y; 5400 = 34 x + 67 y"""
    def test_task1(self):
        m, v = elftasks.parse_claw_machine(self.data)
        self.assertEqual([94, 22], list(m[0]))
        self.assertEqual([34, 67], list(m[1]))
        self.assertEqual([8400, 5400], list(v))

    data2 = """Button A: X+26, Y+66
Button B: X+67, Y+21
Prize: X=12748, Y=12176""".split('\n')

    data3 = """Button A: X+17, Y+86
Button B: X+84, Y+37
Prize: X=7870, Y=6450""".split('\n')

    data4 = """Button A: X+69, Y+23
Button B: X+27, Y+71
Prize: X=18641, Y=10279""".split('\n')

    def test_no_solution(self):
        m, v = elftasks.parse_claw_machine(self.data2)
        m, v = elftasks.parse_claw_machine(self.data3)
        m, v = elftasks.parse_claw_machine(self.data4)



###############


class TestDay14(unittest.TestCase):
    data = """p=0,4 v=3,-3
p=6,3 v=-1,-3
p=10,3 v=-1,2
p=2,0 v=2,-1
p=0,0 v=1,3
p=3,0 v=-2,-2
p=7,6 v=-1,-3
p=3,0 v=-1,-2
p=9,3 v=2,3
p=7,3 v=-1,2
p=2,4 v=2,-3
p=9,5 v=-3,-3""".split('\n')
    def test_task1(self):
        self.assertEqual(((0, 4), (3, -3)), elftasks.parse_robot(self.data[0]))

        robots = [elftasks.parse_robot(r) for r in self.data]
        self.assertEqual((1, 3), tuple(elftasks.move_robot(5, (11, 7), (2, 4), (2, -3))))


###############


class TestDay15(unittest.TestCase):
    data = """########
#..O.O.#
##@.O..#
#...O..#
#.#.O..#
#...O..#
#......#
########

<^^>>>vv<v>>v<<""".split('\n')
    def test_task1(self):
        warehouse = elftasks.Warehouse(self.data[:8])
        self.assertEqual((2, 2), warehouse.robot)
        self.assertEqual((2, 2), warehouse.move_robot("<"))
        self.assertEqual((1, 2), warehouse.move_robot("^"))
        self.assertEqual((1, 2), warehouse.move_robot("^"))

        self.assertEqual((1, 3), warehouse.move_robot(">"))
        self.assertEqual('.', warehouse.changes[(1,3)])
        self.assertEqual('O', warehouse.changes[(1,4)])
        self.assertEqual((1, 4), warehouse.move_robot(">"))
        self.assertEqual('.', warehouse.changes[(1,4)])
        self.assertEqual('O', warehouse.changes[(1,6)])

        self.assertEqual((1, 4), warehouse.move_robot(">"))

        self.assertEqual((2, 4), warehouse.move_robot("v"))
        self.assertEqual('.', warehouse.changes[(2, 4)])
        self.assertEqual('O', warehouse.changes[(6, 4)])

        changes = warehouse.changes.copy()

        self.assertEqual((2, 4), warehouse.move_robot("v"))
        self.assertEqual((2, 3), warehouse.move_robot("<"))
        self.assertEqual((3, 3), warehouse.move_robot("v"))
        self.assertEqual(changes, warehouse.changes)

        self.assertEqual((3, 4), warehouse.move_robot(">"))
        self.assertEqual('.', warehouse.changes[(3, 4)])
        self.assertEqual('O', warehouse.changes[(3, 5)])

        self.assertEqual((3, 5), warehouse.move_robot(">"))
        self.assertEqual('.', warehouse.changes[(3, 5)])
        self.assertEqual('O', warehouse.changes[(3, 6)])

        self.assertEqual((4, 5), warehouse.move_robot("v"))

        self.assertEqual((4, 4), warehouse.move_robot("<"))
        self.assertEqual('.', warehouse.changes[(4, 4)])
        self.assertEqual('O', warehouse.changes[(4, 3)])
        self.assertEqual((4, 4), warehouse.robot)

    def test_hyper_warehouse(self):
        warehouse = elftasks.HyperWarehouse(self.data[:8])

        self.assertEqual(((1, 6), (1, 7)), warehouse.get_box_boundary((1, 6)))
        self.assertEqual(((1, 6), (1, 7)), warehouse.get_box_boundary((1, 7)))

        self.assertEqual([(1, 6), (1, 7)], warehouse.can_move((1,6), (0, 1)))
        self.assertEqual([(1, 6), (1, 7)], warehouse.can_move((1, 6), (0, 1)))
        self.assertEqual([(2, 8), (2, 9), (3, 8), (3, 9), (4, 8), (4, 9), (5, 8), (5, 9)],
                         warehouse.can_move((2,8), (1, 0)))


    def test_cascading_move(self):
        data = """##########
#..O..O.O#
#......O.#
#.OO..O.O#
#..O@..O.#
#O#..O...#
#O..O..O.#
#.OO.O.OO#
#....O...#
##########""".split('\n')

        warehouse = elftasks.HyperWarehouse(data)
        warehouse.move_box(((6, 14), (6, 15)), (0, 1))
        self.assertEqual([(6, 15), (6, 16), (7, 14), (7, 15), (7, 16), (7, 17)], warehouse.can_move((6,15), (1, 0)))

    def test_hyper_gps(self):
        data = """##########
#..O..O.O#
#......O.#
#.OO..O.O#
#..O@..O.#
#O#..O...#
#O..O..O.#
#.OO.O.OO#
#....O...#
##########""".split('\n')

        move_data = """<vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^
vvv<<^>^v^^><<>>><>^<<><^vv^^<>vvv<>><^^v>^>vv<>v<<<<v<^v>^<^^>>>^<v<v
><>vv>v^v^<>><>>>><^^>vv>v<^^^>>v^v^<^^>v^^>v^<^v>v<>>v^v^<v>v^^<^^vv<
<<v<^>>^^^^>>>v^<>vvv^><v<<<>^^^vv^<vvv>^>v<^^^^v<>^>vvvv><>>v^<<^^^^^
^><^><>>><>^^<<^^v>>><^<v>^<vv>>v>>>^v><>^v><<<<v>>v<v<v>vvv>^<><<>^><
^>><>^v<><^vvv<^^<><v<<<<<><^v<<<><<<^^<v<^^^><^>>^<v^><<<^>>^v<v^v<v^
>^>>^v>vv>^<<^v<>><<><<v<<v><>v<^vv<<<>^^v^>^^>>><<^v>>v^v><^^>>^<>vv^
<><^^>^^^<><vvvvv^v<v<<>^v<v>v<<^><<><<><<<^^<<<^<<>><<><^^^>^^<>^>v<>
^^>vv<^v^v<vv>^<><v<^v>^^^>>>^^vvv^>vvv<>>>^<^>>>>>^<<^v>^vvv<>^<><<v>
v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^""".split('\n')

        moves = []
        for line in move_data:
            moves.extend(line)

        warehouse = elftasks.HyperWarehouse(data)
        warehouse.move_robot(moves)
        self.assertEqual(9021, warehouse.gps_coords())


###############


class TestDay16(unittest.TestCase):
    data = """###############
#.......#....E#
#.#.###.#.###.#
#.....#.#...#.#
#.###.#####.#.#
#.#.#.......#.#
#.#.#####.###.#
#...........#.#
###.#.#####.#.#
#...#.....#.#.#
#.#.#.###.#.#.#
#.....#...#.#.#
#.###.#.#.#.#.#
#S..#.....#...#
###############""".split('\n')
    def test_task1(self):
        maze = elftasks.Maze(self.data)
        self.assertEqual(((13, 1), (1, 13)), (maze.start, maze.end))
        self.assertEqual(7036, min([path[-1][2] for path in maze.paths]))

    def test_maze(self):
        data = """#################
#...#...#...#..E#
#.#.#.#.#.#.#.#.#
#.#.#.#...#...#.#
#.#.#.#.###.#.#.#
#...#.#.#.....#.#
#.#.#.#.#.#####.#
#.#...#.#.#.....#
#.#.#####.#.###.#
#.#.#.......#...#
#.#.###.#####.###
#.#.#...#.....#.#
#.#.#.#####.###.#
#.#.#.........#.#
#.#.#.#########.#
#S#.............#
#################""".split('\n')
        maze = elftasks.Maze(data)
        self.assertEqual(11048, min([path[-1][2] for path in maze.paths]))

        min_paths = [path for path in maze.paths if path[-1][2] == 11048]
        tiles = set()
        for path in min_paths:
            [tiles.add(t[0]) for t in path]

#        self.assertEqual(64, len(tiles))


###############


class TestDay17(unittest.TestCase):
    def test_small(self):
        debugger = elftasks.Debugger(0, 0, 9, [2,6])
        debugger.run()
        self.assertEqual(1, debugger.B)

        debugger = elftasks.Debugger(0, 29, 0, [1, 7])
        debugger.run()
        self.assertEqual(26, debugger.B)

        debugger = elftasks.Debugger(0, 2024, 43690, [4, 0])
        debugger.run()
        self.assertEqual(44354, debugger.B)

        debugger = elftasks.Debugger(10, 0, 0, [5,0,5,1,5,4])
        debugger.run()
        self.assertEqual([0, 1, 2], debugger.output)

    def test_task1(self):
        debugger = elftasks.Debugger(729, 0, 0, [0,1,5,4,3,0])
        debugger.run()
        self.assertEqual([4,6,3,5,6,3,5,2,1,0], debugger.output)

    def test_task(self):
        debugger = elftasks.Debugger(2024, 0, 0, [0,1,5,4,3,0])
        debugger.run()
        self.assertEqual([4,2,5,6,7,7,7,7,3,1,0], debugger.output)
        self.assertEqual(0, debugger.A)

    def test_task2(self):
        debugger = elftasks.Debugger(117440, 0, 0, [0,1,5,4,3,0], [0,1,5,4,3,0])
        debugger.run()
#        self.assertEqual([0,1,5,4,3,0], debugger.output)


###############
import re
import functools

class TestDay19(unittest.TestCase):
    towels = """r, wr, b, g, bwu, rb, gb, br""".split(', ')
    designs = """brwrr
bggr
gbbr
rrbgbr
ubwu
bwurrg
brgr
bbrgwb""".split('\n')
    def test_task1(self):
        designer = elftasks.TowelDesignMatchingController(self.towels, self.designs)
        matches = designer.match_designs()
        self.assertEqual(6, sum(matches))

    def test_task2(self):
        designer = elftasks.TowelDesignMatchingController(self.towels, self.designs)
        matches = designer.match_designs()
#        print(designer.design_catalogue[self.designs[0]])
#        self.assertEqual(2, designer.count_designs(self.designs[0]))
#        counts = [designer.count_designs(design) for design in self.designs]
#        self.assertEqual(16, sum(counts))


###############


class TestDay21(unittest.TestCase):
    data = """""".split('\n')
    def test_task1(self):
        self.assertEqual(False, False)

###############


class TestDay23(unittest.TestCase):
    data = [x.split('-') for x in """kh-tc
qp-kh
de-cg
ka-co
yn-aq
qp-ub
cg-tb
vc-aq
tb-ka
wh-tc
yn-cg
kh-ub
ta-co
de-co
tc-td
tb-wq
wh-td
ta-ka
td-qp
aq-cg
wq-ub
ub-vc
de-ta
wq-aq
wq-vc
wh-yn
ka-de
kh-ta
co-tc
wh-qp
tb-vc
td-yn""".split('\n')]
    def test_task1(self):
        expected = set([tuple(sorted(x.split(','))) for x in """aq,cg,yn
aq,vc,wq
co,de,ka
co,de,ta
co,ka,ta
de,ka,ta
kh,qp,ub
qp,td,wh
tb,vc,wq
tc,td,wh
td,wh,yn
ub,vc,wq""".split('\n')])

        rings = elftasks.find_rings(elftasks.create_lan(self.data))
        self.assertEqual(12, len(rings))
        self.assertEqual(expected, rings)
###############


class TestDay24(unittest.TestCase):
    wires = """x00: 1
x01: 1
x02: 1
y00: 0
y01: 1
y02: 0""".split('\n')

    gates = """x00 AND y00 -> z00
x01 XOR y01 -> z01
x02 OR y02 -> z02""".split('\n')

    def test_task1(self):
        wires = elftasks.parse_wires(self.wires)
        self.assertEqual({'x00': 1, 'x01': 1, 'x02': 1, 'y00': 0, 'y01': 1, 'y02': 0}, wires)
        self.assertEqual([(['x00', 'AND', 'y00'], 'z00'), (['x01', 'XOR', 'y01'], 'z01'), (['x02', 'OR', 'y02'], 'z02')], elftasks.parse_gates(self.gates))

        for gate in elftasks.parse_gates(self.gates):
            elftasks.run_gate(gate, wires)
        self.assertEqual(0, wires['z00'])
        self.assertEqual(0, wires['z01'])
        self.assertEqual(1, wires['z02'])

###############


class TestDay25(unittest.TestCase):
    data = """#####
.####
.####
.####
.#.#.
.#...
.....

#####
##.##
.#.##
...##
...#.
...#.
.....

.....
#....
#....
#...#
#.#.#
#.###
#####

.....
.....
#.#..
###..
###.#
###.#
#####

.....
.....
.....
#....
#.#..
#.#.#
#####""".split('\n')
    def test_task1(self):
        locks, keys = elftasks.parse_lock_keys(self.data)
        print(locks, keys)
        self.assertEqual([0, 5, 3, 4, 3], locks[0])
        self.assertEqual([5, 0, 2, 1, 3], locks[0])