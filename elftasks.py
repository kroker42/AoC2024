import re
import time
import math
from collections import Counter
from itertools import pairwise


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
    