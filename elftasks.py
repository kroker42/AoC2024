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
        print(self.map[(0,0)])
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
    