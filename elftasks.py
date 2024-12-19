import re
import time
import math
import operator

import numpy
import numpy as np
from collections import Counter
from itertools import pairwise
from itertools import combinations


##############
def sum_min_distances(lists):
    sorted_lists = [sorted(l) for l in lists]
    distances = [abs(p[0] - p[1]) for p in zip(*sorted_lists)]
    return sum(distances)


def similarity_score(lists):
    frequencies = [Counter(list(l)) for l in lists]
    scores = [key * frequencies[0][key] * frequencies[1][key] for key in frequencies[0].keys()]
    return sum(scores)


def day1():
    data = list(zip(*[line.split() for line in open('input01.txt')]))
    data = [list(map(int, l)) for l in data]
    start_time = time.time()

    task1 = sum_min_distances(data)
    task2 = similarity_score(data)

    return time.time() - start_time, task1, task2
    

##############

def diff_levels(report):
    return [x - y for x, y in pairwise(report)]

def is_safe_diff(diff):
    return all(0 < x <= 3 for x in diff) or all(-3 <= x < 0 for x in diff)

def is_safe(report):
    return is_safe_diff(diff_levels(report))

def dampen_report(report):
    for i in range(len(report)):
        if is_safe(report[0:i] + report[i+1:]):
            return True

    return is_safe(report)


def day2():
    reports = [list(map(int, line.split())) for line in open('input02.txt')]
    start_time = time.time()

    task1 = sum([is_safe(r) for r in reports])
    task2 = sum([dampen_report(r) for r in reports])

    return time.time() - start_time, task1, task2
    

##############

def multiply_re(instructions):
    return re.findall(r'mul\((\d+),(\d+)\)', instructions)

def multiply(num_pairs):
    return [int(x) * int(y) for x,y in num_pairs]

def do_re(instructions):
    return re.findall(r'do\(\)|don\'t\(\)|mul\(\d+,\d+\)', instructions)

def parse_instructions(instructions):
    on_off = True
    sum_prod = 0

    for instr in do_re(instructions):
        match instr:
            case "do()":
                on_off = True
            case "don't()":
                on_off = False
            case _:
                if on_off:
                    sum_prod += multiply(multiply_re(instr))[0]

    return sum_prod


def day3():
    data = open('input03.txt').read()
    start_time = time.time()

    task1 = sum(multiply(multiply_re(data)))
    task2 = parse_instructions(data)

    return time.time() - start_time, task1, task2
    

##############

class Grid:
    def __init__(self, grid):
        self.grid = grid
        self.row_range = range(len(grid))
        self.col_range = range(len(grid[0]))

    def get(self, row, col):
        if row in self.row_range and col in self.col_range:
            return self.grid[row][col]
        return ''

    def count(self, count_fn):
        count = 0
        for row in self.row_range:
            for col in self.col_range:
                count += count_fn(self, row, col)
        return count


def count_xmas(xmas_grid, row, col):
    count = 0

    for r in [0, 1, -1]:
        for c in [0, 1, -1]:
            xmas = xmas_grid.get(row, col)
            for i in range(1, 4):
                xmas += xmas_grid.get(row + i * r, col + i * c)
            count += xmas == "XMAS"

    return count


def count_x_mas(xmas_grid, row, col):
    count = xmas_grid.get(row - 1, col -1) + xmas_grid.get(row, col) + xmas_grid.get(row + 1, col + 1) in ["MAS", "SAM"]
    count += xmas_grid.get(row + 1, col -1) + xmas_grid.get(row, col) + xmas_grid.get(row - 1, col + 1) in ["MAS", "SAM"]
    return count == 2


def day4():
    data = [line.strip() for line in open('input04.txt')]
    start_time = time.time()

    grid = Grid(data)

    task1 = grid.count(count_xmas)
    task2 = grid.count(count_x_mas)

    return time.time() - start_time, task1, task2
    

##############
class PrintRules:
    def __init__(self, data):
        self.rules, self.print_runs = self.parse_printing_rules(data)

    def parse_printing_rules(self, rules):
        ordering = {}
        print_runs = []

        for rule in rules:
            if len(rule) < 5:
                continue

            if rule[2] == '|':
                order = [int(x) for x in rule.split('|')]
                if order[0] in ordering:
                    ordering[order[0]].append(order[1])
                else:
                    ordering[order[0]] = [order[1]]
            else:
                print_runs.append([int(x) for x in rule.split(',')])

        return ordering, print_runs

    def validate_print_run(self, pages):
        for i in reversed(range(1, len(pages))):
            for j in range(i):
                if pages[i] in self.rules and pages[j] in self.rules[pages[i]]:
                    return (j, i)
        return (0, 0)

    def get_run_mid_pages(self, print_runs):
        return [print_run[len(print_run)//2] for print_run in print_runs if self.validate_print_run(print_run) == (0, 0)]

    def get_mid_pages(self):
        return self.get_run_mid_pages(self.print_runs)

    def reorder(self, print_run):
        i, j = self.validate_print_run(print_run)
        while i != j:
            print_run[i], print_run[j] = print_run[j], print_run[i]
            i, j = self.validate_print_run(print_run)
        return print_run


    def reorder_print_runs(self):
        reordered_runs = []
        for print_run in self.print_runs:
            if self.validate_print_run(print_run) != (0, 0):
                reordered_runs.append(self.reorder(print_run))
        return reordered_runs



def day5():
    data = [line.strip() for line in open('input05.txt')]
    start_time = time.time()

    rule_book = PrintRules(data)

    task1 = sum(rule_book.get_mid_pages())
    task2 = sum(rule_book.get_run_mid_pages(rule_book.reorder_print_runs()))

    return time.time() - start_time, task1, task2
    

##############

def get_map(lines):
    obstacles = []
    guard = (0, 0)
    for r in range(len(lines)):
        for c in range(len(lines[0])):
            if lines[r][c] == '^':
                return r, c

def is_valid_coords(pos, data):
    return 0 <= pos[0] < len(data) and 0 <= pos[1] < len(data[0])

def find_obstacle(start, direction, data):
    visited = {tuple(start)}
    pos = np.add(start, direction)
    while is_valid_coords(pos, data) and data[pos[0]][pos[1]] != '#':
        visited.add(tuple(pos))
        pos = np.add(pos, direction)

    return pos.tolist(), visited

def path_length(guard, data):
    next_direction = {(-1, 0): (0, 1), (0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0)}

    direction = (-1, 0)
    obstacle, path = find_obstacle(guard, direction, data)
    visited = {*path}

    while is_valid_coords(obstacle, data):
        guard = numpy.subtract(obstacle, direction)
        direction = next_direction[direction]
        obstacle, path = find_obstacle(guard, direction, data)
        visited |= path

    return visited

def find_added_obstacle(start, direction, data, new_obstacle_pos):
    pos = np.add(start, direction)
    while is_valid_coords(pos, data) and (tuple(pos) != new_obstacle_pos and data[pos[0]][pos[1]] != '#'):
        pos = np.add(pos, direction)

    return tuple(pos)


def detect_loop(guard, data, new_obstacle_pos):
    next_direction = {(-1, 0): (0, 1), (0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0)}

    direction = (-1, 0)
    obstacle = find_added_obstacle(guard, direction, data, new_obstacle_pos)
    visited_obstacles = {obstacle: [direction]}

    while is_valid_coords(obstacle, data):
        guard = numpy.subtract(obstacle, direction)
        direction = next_direction[direction]
        obstacle = find_added_obstacle(guard, direction, data, new_obstacle_pos)
        if obstacle in visited_obstacles:
            if direction in visited_obstacles[obstacle]:
                return 1
            else:
                visited_obstacles[obstacle].append(direction)
        else:
            visited_obstacles[obstacle] = [direction]
    return 0


def add_obstacles(guard, visited, data):
    loops = 0
    for pos in visited:
        loops += detect_loop(guard, data, pos)

    return loops


def day6():
    data = [line.strip() for line in open('input06.txt')]
    start_time = time.time()

    guard = get_map(data)
    path = path_length(guard, data)
    task1 = len(path)

    task2 = add_obstacles(guard, path, data)

    return time.time() - start_time, task1, task2
    

##############

def get_equations(data):
    equations = [eq.split(':') for eq in data]
    equations = [(int(eq[0]), [int(x)  for x in eq[1].strip().split(' ')]) for eq in equations]
    return equations


class EvaluationTree:

    def __init__(self, result, equation):
        self.result = result
        self.equation = equation
        self.root = [[equation[0]]]


    def evaluate(self, ops):
        for val in self.equation[1:]:
            results = []
            for leaf in self.root[-1]:
                results.extend([op(leaf, val) for op in ops])
            self.root.append(results)

        return self.result in self.root[-1]

def concat_nums(x, y):
    return int(str(x) + str(y))

def day7():
    equations = get_equations(open('input07.txt'))
    start_time = time.time()

    results = []
    for eq in equations:
        if EvaluationTree(eq[0], eq[1]).evaluate([operator.mul, operator.add]):
            results.append(eq[0])
    task1 = sum(results)

    results = []
    for eq in equations:
        if EvaluationTree(eq[0], eq[1]).evaluate([operator.mul, operator.add, concat_nums]):
            results.append(eq[0])
    task2 = sum(results)

    return time.time() - start_time, task1, task2
        



##############

def find_masts(grid):
    masts = {}
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != '.':
                if grid[r][c] in masts:
                    masts[grid[r][c]].append((r, c))
                else:
                    masts[grid[r][c]] = [(r, c)]
    return masts


def find_antinodes(masts):
    antinodes = set()
    for positions in masts.values():
        pairs = combinations(positions, 2)
        for p in pairs:
            dist = np.subtract(*p)
            antinodes.add(tuple(np.add(p[0], dist)))
            antinodes.add(tuple(np.add(p[1], -dist)))

    return antinodes

def nodes_in_grid(nodes, grid):
    return [n for n in nodes if is_valid_coords(n, grid)]

def find_harmonic_antinodes(masts, grid):
    antinodes = set()
    for positions in masts.values():
        if len(positions) > 2:
            antinodes.update(positions)

        pairs = combinations(positions, 2)
        for p in pairs:
            dist = np.subtract(*p)

            node = np.add(p[0], dist)
            while is_valid_coords(node, grid):
                antinodes.add(tuple(node))
                node = np.add(node, dist)

            node = np.add(p[1], -dist)
            while is_valid_coords(node, grid):
                antinodes.add(tuple(node))
                node = np.add(node, -dist)

    return antinodes

def day8():
    data = [line.strip() for line in open('input08.txt')]
    start_time = time.time()

    masts = find_masts(data)
    task1 = len(nodes_in_grid(find_antinodes(masts), data))
    task2 = len(find_harmonic_antinodes(masts, data))

    return time.time() - start_time, task1, task2
    

##############

def fragment_disk(disk):
    filesystem = []

    files = disk[0::2]
    spaces = disk[1::2]

    last_file = len(files) - 1

    curr_index = 0

    while curr_index <= last_file:
        filesystem.extend(files[curr_index] * [curr_index]) # e.g. fs += '000'

        while spaces[curr_index] > 0 and last_file > curr_index:
            if files[last_file] > 0:
                no_blocks = min(spaces[curr_index], files[last_file])
                filesystem.extend(no_blocks * [last_file])
                spaces[curr_index] -= no_blocks
                files[last_file] -= no_blocks

            if files[last_file] == 0:
                last_file -= 1

        curr_index += 1

    return filesystem

def compress_disk(disk):
    filesystem = []

    files = list(enumerate(disk[0::2]))
    spaces = disk[1::2]

    for curr_index in range(len(files) - 1):
        filesystem.extend(files[curr_index][1] * [files[curr_index][0]]) # e.g. fs += '000'

        while spaces[curr_index] > 0:
            rev_list = reversed(files[curr_index + 1:])
            file_num, file_size = next((x for x in rev_list if x[0] > 0 and 0 < x[1] <= spaces[curr_index]), (0, spaces[curr_index]))
            filesystem.extend(file_size * [file_num])
            if file_num > 0:
                files[file_num] = (0, file_size)
                spaces[curr_index] -= file_size
            else:
                spaces[curr_index] = 0

    filesystem.extend(files[-1][1] * [files[-1][0]]) # e.g. fs += '000'

    return filesystem

def checksum(filesystem):
    return sum([k * v for k, v in enumerate(filesystem)])

def day9():
    data = [line.strip() for line in open('input09.txt')]
    disk = [int(x) for x in data[0]]
    start_time = time.time()

    task1 = checksum(fragment_disk(disk))
    task2 = checksum(compress_disk(disk))

    return time.time() - start_time, task1, task2
    

##############

class IslandMap:
    def __init__(self, map):
        self.map = np.array(map)
        self.paths = {}
        self.elevations, self.trails = self.index()

    def index(self):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elevations = {i : [] for i in range(10)}
        trails = {}
        for r in range(len(self.map)):
            for c in range(len(self.map[0])):
                coords = (r, c)
                elevation = self.map[coords]
                elevations[int(elevation)].append(coords)

                for dir in directions:
                    neighbour = np.add(coords, dir)
                    if is_valid_coords(neighbour, self.map) and self.map[tuple(neighbour)] == elevation - 1:
                        if coords in trails:
                            trails[coords].append(tuple(neighbour))
                        else:
                            trails[coords] = [tuple(neighbour)]

        return elevations, trails

    def score_trails(self):
        scores = {coords : {coords} for coords in self.elevations[9]}

        points = set(self.elevations[9])
        while points:
            next_points = set()
            for p in points:
                if p in self.trails:
                    neighbours = self.trails[p]
                    next_points.update(neighbours)
                    for n in neighbours:
                        if n in scores:
                            scores[n].update(scores[p])
                        else:
                            scores[n] = scores[p].copy()

            points = next_points

        return sum([len(scores[i]) for i in self.elevations[0] if i in scores])

    def score_distinct_trails(self):
        scores = {coords : 1 for coords in self.elevations[9]}

        points = set(self.elevations[9])
        while points:
            next_points = set()
            for p in points:
                if p in self.trails:
                    neighbours = self.trails[p]
                    next_points.update(neighbours)
                    for n in neighbours:
                        if n in scores:
                            scores[n] += scores[p]
                        else:
                            scores[n] = scores[p]

            points = next_points

        return sum([scores[i] for i in self.elevations[0] if i in scores])


def day10():
    data = [[int(x) for x in line.strip()] for line in open('input10.txt')]
    start_time = time.time()

    map = IslandMap(data)
    task1 = map.score_trails()
    task2 = map.score_distinct_trails()

    return time.time() - start_time, task1, task2
    

##############

def blink(stones):
    new_stones = []
    for stone in stones:
        if stone == 0:
            new_stones.append(1)
        elif len(str(stone)) % 2 == 0:
            nums = str(stone)
            new_stones.append(int(nums[:len(nums)//2]))
            new_stones.append(int(nums[len(nums)//2:]))
        else:
            new_stones.append(stone * 2024)
    return new_stones

def count_stones(stones):
    stone_buckets = {}
    for stone in stones:
        if stone in stone_buckets:
            stone_buckets[stone] += 1
        else:
            stone_buckets[stone] = 1
    return stone_buckets

def update(buckets, stone, count):
    if stone in buckets:
        buckets[stone] += count
    else:
        buckets[stone] = count

def blink_at_buckets(stone_buckets):
    buckets = {}
    for stone in stone_buckets:
        if stone == 0:
            update(buckets, 1, stone_buckets[0])
        elif len(str(stone)) % 2 == 0:
            nums = str(stone)
            update(buckets, int(nums[:len(nums)//2]), stone_buckets[stone])
            update(buckets, int(nums[len(nums)//2:]), stone_buckets[stone])
        else:
            update(buckets, stone * 2024, stone_buckets[stone])
    return buckets

def day11():
    original_stones = [int(x) for line in open('input11.txt') for x in line.strip().split(' ')]
    start_time = time.time()

    stones = original_stones
    for i in range(25):
        stones = blink(stones)

    task1 = len(stones)

    buckets = count_stones(original_stones)
    for i in range(75):
        buckets = blink_at_buckets(buckets)
    task2 = sum(buckets.values())

    return time.time() - start_time, task1, task2
    

##############

class GardenPatch:
    def __init__(self, patch):
        self.patch = np.array(patch)
        self.regions = []
        self.fenced_off = {}

    # def categorise(self):
    #     for row in range(len(self.patch)):
    #         for col in range(len(self.patch[0])):
    #             self.areas.setdefault(self.patch[row][col], []).append((row, col))


    def find_neighbours(self, coords):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        neighbours = []
        for dir in directions:
            n = np.add(coords, dir)
            if is_valid_coords(n, self.patch) and self.patch[*n] == self.patch[*coords]:
                neighbours.append(tuple(n))

        return neighbours

    def find_regions(self):
        for row in range(len(self.patch)):
            for col in range(len(self.patch[0])):
                coords = (row, col)
                if coords not in self.fenced_off:
                    neighbours = self.find_neighbours(coords)
                    self.fenced_off[coords] = neighbours.copy()
                    region = {coords}

                    i = 0
                    while i < len(neighbours):
                        if tuple(neighbours[i]) not in self.fenced_off:
                            new_neighbours = self.find_neighbours(neighbours[i])
                            self.fenced_off[tuple(neighbours[i])] = new_neighbours.copy()
                            neighbours.extend(new_neighbours)
                            region.add(tuple(neighbours[i]))
                        i += 1

                    self.regions.append(region)

    def is_outie_corner(self, coord):
        return self.fenced_off[coord][0][0] != self.fenced_off[coord][1][0] and \
               self.fenced_off[coord][0][1] != self.fenced_off[coord][1][1]

    def is_innie_corner(self, coord, region):
        diagonals = [(-1, -1), (1, 1), (1, -1), (-1, 1)]

        no_corners = 0

        for diagonal in diagonals:
            if tuple(np.add(coord, (diagonal[0], 0))) in self.fenced_off[coord] and \
                tuple(np.add(coord, (0, diagonal[1]))) in self.fenced_off[coord]:
                no_corners += tuple(np.add(coord, diagonal)) not in region

        return no_corners

    def find_sides(self, region):
        coords = sorted(region)
        sides = 0

        for coord in coords:
            no_nbrs = len(self.fenced_off[coord])
            if no_nbrs == 0:
                sides += 4
            elif no_nbrs == 1:
                sides += 2
            elif no_nbrs == 2:
                sides += self.is_outie_corner(coord)
                sides += self.is_innie_corner(coord, region)
            else:
                sides += self.is_innie_corner(coord, region)

        return sides


def day12():
    data = [list(line.strip()) for line in open('input12.txt')]
    start_time = time.time()

    patch = GardenPatch(data)
    patch.find_regions()

    prices = []
    for region in patch.regions:
        fence = 0
        for coords in region:
            fence += 4 - len(patch.fenced_off[coords])
        prices.append(len(region) * fence)

    task1 = sum(prices)

    prices = []
    for region in patch.regions:
        sides = patch.find_sides(region)
        prices.append(len(region) * sides)

    task2 = sum(prices)

    return time.time() - start_time, task1, task2
    

##############

"""Button A: X+94, Y+34
Button B: X+22, Y+67
Prize: X=8400, Y=5400"""

def parse_button(data):
    eq = data.split('+')
    return [int(eq[1].split(',')[0]), int(eq[2])]

def parse_prize(data):
    eq = data.split('=')
    return [int(eq[1].split(',')[0]), int(eq[2])]

def parse_claw_machine(data):
    button_a = parse_button(data[0])
    button_b = parse_button(data[1])
    m = np.array([[button_a[0], button_b[0]], [button_a[1], button_b[1]]])
    v = np.array(parse_prize(data[2]))
    return m, v



def day13():
    data = [line.strip() for line in open('input13.txt')]
    start_time = time.time()

    claw_machines = []
    for i in range(0, len(data), 4):
        claw_machines.append(parse_claw_machine(data[i: i + 3]))

    moves = []
    for machine in claw_machines:
        moves.append(np.linalg.solve(*machine))

    moves = [x for x in moves if abs(x[0] - np.rint(x[0])) < 0.001 and abs(x[1] - np.rint(x[1])) < 0.001 and 0 <= x[0] <= 100 and 0 <= x[1] <= 100]

    task1 = sum([3 * x[0] + x[1] for x in moves])

    moves = []
    for machine in claw_machines:
        moves.append(np.linalg.solve(machine[0], np.add(machine[1], np.array([10000000000000, 10000000000000]))))

    moves = [x for x in moves if abs(x[0] - np.rint(x[0])) < 0.001 and abs(x[1] - np.rint(x[1])) < 0.001 and 0 <= x[0] and 0 <= x[1]]
    task2 = sum([3 * x[0] + x[1] for x in moves])

    return time.time() - start_time, task1, task2
    

##############

"""p=6,3 v=-1,-3"""
def parse_robot(position):
    bits = position.split('=')
    position = tuple([int(x) for x in bits[1].split(' ')[0].split(',')])
    velocity = tuple([int(x) for x in bits[2].split(',')])
    return position, velocity

def move_robot(time, space, position, velocity):
    return np.mod(np.add(position, time * np.array(velocity)), space)

def count_bots(quadrant, bots):
    quad_bots = [bot for bot in bots if quadrant[0][0] <= bot[0] < quadrant[1][0] and quadrant[0][1] <= bot[1] < quadrant[1][1]]
    return len(quad_bots)

def safety_factor(bots, quads):
    return math.prod([count_bots(q, bots) for q in quads])

def get_quadrants(space):
    return [[(0, 0), (space[0] // 2, space[1] // 2)], [(0, space[1] // 2 + 1), (space[0] // 2, space[1])],
            [(space[0] // 2 + 1, 0), (space[0], space[1] // 2)], [(space[0] // 2 + 1, space[1] // 2 + 1), space]]


def print_robots(robots, space):
    positions = {}
    for r in robots:
        if tuple(r) in positions:
            positions[tuple(r)] += 1
        else:
            positions[tuple(r)] = 1

    for row in range(space[1]):
        line = ""
        for col in range(space[0]):
            line += '.' if (col, row) not in positions else str(positions[(col, row)])
        print(line)


def day14():
    data = [line.strip() for line in open('input14.txt')]
    start_time = time.time()

    robots = [parse_robot(r) for r in data]
    space = (101, 103)
    moved_bots = [move_robot(100, space, *r) for r in robots]
    quadrants = get_quadrants(space)
    task1 = safety_factor(moved_bots, quadrants)

    for i in range(4000, 10000):
        moved_bots = [tuple(move_robot(i, space, *r)) for r in robots]
        if len(set(moved_bots)) == len(robots):
            print(i)
            print_robots(moved_bots, space)

    task2 = 6771

    return time.time() - start_time, task1, task2
    

##############
"""
########
#..O.O.#
##@.O..#
#...O..#
#.#.O..#
#...O..#
#......#
########
"""

class Warehouse:
    def __init__(self, data):
        self.data = data
        self.robot = self.find_robot()
        self.changes = {} # (4, 5) : 'O'

    def find_robot(self):
        for i in range(len(self.data)):
            if '@' in self.data[i]:
                return i, self.data[i].index('@')
        return False

    def print(self):
        for r in range(len(self.data)):
            row = list(self.data[r])
            for c in range(len(self.data[0])):
                if (r, c) in self.changes:
                    row[c] = self.changes[(r, c)]
                if (r, c) == self.robot:
                    row[c] = '@'
            print("".join(row))

    def is_a(self, pos, obj_type):
        return (pos in self.changes and self.changes[pos] == obj_type) or \
            (pos not in self.changes and self.data[pos[0]][pos[1]] == obj_type)

    def has_box(self, pos):
        return self.is_a(pos, 'O')

    def is_empty(self, pos):
        return self.is_a(pos, '.') or self.is_a(pos, '@')

    def is_wall(self, pos):
        return self.is_a(pos, '#')

    def move(self, pos, direction):
        new_pos = tuple(np.add(pos, direction))
        if not is_valid_coords(new_pos, self.data):
            raise ValueError(new_pos)
        return new_pos

    def move_boxes(self, pos, direction):
        if not self.has_box(pos):
            return

        new_pos = pos
        while self.has_box(new_pos):
            new_pos = self.move(new_pos, direction)

        if self.is_empty(new_pos):
            self.changes[pos] = '.'
            self.changes[new_pos] = 'O'

    def move_robot(self, moves):
        directions = {'>': np.array([0, 1]), '^': np.array([-1, 0]),
                      'v': np.array([1, 0]), '<': np.array([0, -1])}

        for move in moves:
            new_pos = self.move(self.robot, directions[move])
            self.move_boxes(new_pos, directions[move])
            if self.is_empty(new_pos):
                self.robot = new_pos

        return self.robot

    def gps_coords(self):
        gps = []
        for r in range(len(self.data)):
            for c in range(len(self.data[0])):
                if self.has_box((r, c)):
                    gps.append((r, c))

        return sum([100 * x + y for (x, y) in gps])

def inflate(data):
    new_map = []
    for row in data:
        new_map.append([])
        for ch in row:
            match ch:
                case '.':
                    new_map[-1].extend('..')
                case '@':
                    new_map[-1].extend('@.')
                case '#':
                    new_map[-1].extend('##')
                case 'O':
                    new_map[-1].extend('[]')
                case _:
                    raise ValueError(_)

    return new_map

class HyperWarehouse(Warehouse):
    def __init__(self, data):
        super().__init__(inflate(data))

    def has_box(self, pos):
        return self.is_a(pos, '[') or self.is_a(pos, ']')

    def get_symbol(self, pos):
        if pos in self.changes:
            return self.changes[pos]
        else:
            return self.data[pos[0]][pos[1]]

    def get_box_boundary(self, pos):
        if not self.has_box(pos):
            raise ValueError(pos)

        if self.get_symbol(pos) == '[':
            return pos, tuple(np.add(pos, (0, 1)))
        else:
            return tuple(np.add(pos, (0, -1))), pos

    def can_move(self, pos, direction):
        if not self.has_box(pos):
            raise ValueError(pos)

        result = []

        boxes_to_move = list(self.get_box_boundary(pos))
        while boxes_to_move:
            new_boxes_to_move = []

            for box in boxes_to_move:
                new_pos = tuple(np.add(box, direction))
                if new_pos not in boxes_to_move and new_pos not in new_boxes_to_move:
                    if self.is_wall(new_pos):
                        return False

                    if self.has_box(new_pos):
                        new_boxes_to_move.extend(self.get_box_boundary(new_pos))

            result.extend(boxes_to_move)
            boxes_to_move = new_boxes_to_move.copy()

        return result

    def move_box(self, box, direction):
        # if not self.has_box(box[0]) or not self.has_box(box[1]):
        #     print(self.get_symbol(box[0]), self.get_symbol(box[1]))
        #     raise ValueError(box)

        new_box = tuple(np.add(box[0], direction)), tuple(np.add(box[1], direction))
        for pos in new_box:
            if not is_valid_coords(pos, self.data):
                raise ValueError(pos)

        self.changes[box[0]] = '.'
        self.changes[box[1]] = '.'
        self.changes[new_box[0]] = '['
        self.changes[new_box[1]] = ']'

    def move_robot(self, moves):
        directions = {'>': np.array([0, 1]), '^': np.array([-1, 0]),
                      'v': np.array([1, 0]), '<': np.array([0, -1])}

        for move in moves:
            new_pos = self.move(self.robot, directions[move])

            if self.has_box(new_pos):
                boxes = self.can_move(new_pos, directions[move])
                if boxes:
                    for i in reversed(range(0, len(boxes), 2)):
                        self.move_box((boxes[i], boxes[i+1]), directions[move])

            if self.is_empty(new_pos):
                self.robot = new_pos

        return self.robot

    def gps_coords(self):
        gps = []
        for r in range(len(self.data)):
            for c in range(len(self.data[0])):
                if self.get_symbol((r, c)) == '[':
                    gps.append((r, c))

        return sum([100 * x + y for (x, y) in gps])


def day15():
    data = [line.strip() for line in open('input15.txt')]
    start_time = time.time()

    break_line = data.index('')
    warehouse = Warehouse(data[:break_line])

    moves = []
    for m in data[break_line + 1 :]:
        moves.extend(m)
    warehouse.move_robot(moves)
    task1 = warehouse.gps_coords()

    hyper_warehouse = HyperWarehouse(data[:break_line])
    hyper_warehouse.move_robot(moves)
    task2 = hyper_warehouse.gps_coords()

    return time.time() - start_time, task1, task2
    

##############

class Maze:
    def __init__(self, maze):
        self.maze = maze
        self.start, self.end = self.parse_maze()
        self.paths = []
        self.visited = {self.start: 0}
        self.find_paths()

    def parse_maze(self):
        start = (0, 0)
        end = (0, 0)

        for r in range(len(self.maze)):
            if 'E' in self.maze[r]:
                end = (r, self.maze[r].index('E'))

            if 'S' in self.maze[r]:
                start = (r, self.maze[r].index('S'))

        return start, end

    def find_new_path(self, path, direction, weight):
        neighbour = tuple(np.add(path[-1][0], direction))

        if is_valid_coords(neighbour, self.maze) and \
                self.maze[neighbour[0]][neighbour[1]] != '#' and \
                (neighbour not in self.visited or self.visited[neighbour] >= path[-1][2] + weight):
            self.visited[neighbour] = path[-1][2] + weight
            return path.copy() + [(neighbour, direction, path[-1][2] + weight)]
        else:
            return []

    def find_paths(self):
        paths = [[(self.start, (0, 1), 0)]]

        while paths:
            new_paths = []
            for path in paths:
                directions = [(path[-1][1], 1),
                              (tuple(np.multiply(tuple(reversed(path[-1][1])), -1)), 1001),
                              (tuple(reversed(path[-1][1])), 1001)]
                for direction in directions:
                    new_path = self.find_new_path(path, direction[0], direction[1])
                    if new_path:
                        if new_path[-1][0] == self.end:
                            self.paths.append(new_path)
                        else:
                            new_paths.append(new_path)
            paths = new_paths


def day16():
    data = [line.strip() for line in open('input16.txt')]
    start_time = time.time()

    maze = Maze(data)
    task1 = min([path[-1][2] for path in maze.paths])

    min_paths = [path for path in maze.paths if path[-1][2] == task1]

    tiles = set()
    for path in min_paths:
        [tiles.add(t[0]) for t in path]
    task2 = len(tiles)

    return time.time() - start_time, task1, task2
    

##############

class Debugger:
    def __init__(self, A, B, C, program, exp_output = []):
        self.A = A
        self.B = B
        self.C = C
        self.program = program
        self.exp_output = exp_output
        self.historical_As = set([self.A])
        self.output = []
        self.instruction = 0
        self.operands = {0: (lambda: 0), 1: (lambda: 1), 2: (lambda: 2), 3: (lambda: 3),
                         4: (lambda: self.A),
                         5: (lambda: self.B),
                         6: (lambda: self.C),
                         7: None}

        self.instructions = {0: (lambda x: self.adv(x)),
                             1: (lambda x: setattr(self, 'B', self.B ^ x)),
                             2: (lambda x: setattr(self, 'B', self.operands[x]() % 8)),
                             3: (lambda x: setattr(self, 'instruction', x) if self.A != 0 else None),
                             4: (lambda x: setattr(self, 'B', self.B ^ self.C)),
                             5: (lambda x: self.out(x)),
                             6: (lambda x: setattr(self, 'B', self.A // pow(2, self.operands[x]()))),
                             7: (lambda x: setattr(self, 'C', self.A // pow(2, self.operands[x]())))}

    def adv(self, x):
        new_A = self.A // pow(2, self.operands[x]())
        if new_A in self.historical_As:
            raise ValueError(new_A)

        self.A = new_A
        self.historical_As.add(new_A)

    def out(self, x):
        self.output = self.output + [self.operands[x]() % 8]

    def run(self):
        while self.instruction < len(self.program) - 1:
            i = self.instruction
            self.instructions[self.program[self.instruction]](self.program[i + 1])
#            print(self.instruction, self.A, self.B, self.C, self.output)
            if self.instruction == i: # program didn't jump
                self.instruction += 2


def day17():
#    data = [line.strip().split() for line in open('input17.txt')]
    start_time = time.time()

    A = 50230824
    B = 0
    C = 0
    program = [2,4,1,3,7,5,0,3,1,4,4,7,5,5,3,0]

    debugger = Debugger(A, B, C, program)
    debugger.run()

    task1 = str(debugger.output).replace(' ', '')


    debugger2 = Debugger(0, B, C, program)
    try:
        debugger2.run()
    except Exception:
        pass

    i = 35184372078800
    i = 35184374468875
    while debugger2.output != program:
        debugger2 = Debugger(i, B, C, program)
        try:
            debugger2.run()
        except Exception:
            pass

#        print(i, program, debugger2.output)
        i += 1

    task2 = i

    return time.time() - start_time, task1, task2
    

##############
class TowelDesignMatchingController:
    def __init__(self, towels, designs):
        self.towels = towels
        self.designs = designs
        self.matches = {}

    def match(self, design):
        if design not in self.matches:
            self.matches[design] = self.match_design(design)
        return self.matches[design]

    def match_design(self, design):
        if not design:
            return True

        is_match = False

        for towel in self.towels:
            positions = [occurrence.span() for occurrence in re.finditer(towel, design)]
            # if not positions:
            #     break

            # substrings = [(0, positions[0][0])] + [(x[1], y[0]) for x, y in pairwise(positions)]
            # i = 0
            # while not is_match and i < len(substrings):
            #     is_match = match_design(towels, design[substrings[i][0]:substrings[i][1]])
            #     i += 1

            for position in positions:
                is_match = self.match(design[:position[0]]) and self.match(design[position[1]:])
                if is_match:
                    return True

        return is_match


    def match_designs(self):
        return [self.match_design(design) for design in self.designs]

def day19():
    data = [line.strip() for line in open('input19.txt')]
    start_time = time.time()

    towels = data[0].split(', ')
    designs = data[2:]

    designer = TowelDesignMatchingController(towels, designs)

    matches = designer.match_designs()
    task1 = sum(matches)
    task2 = None

    return time.time() - start_time, task1, task2
    