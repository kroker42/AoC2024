import re
import time
import math
import operator
import itertools
import functools

import numpy as np
from collections import Counter
from itertools import pairwise
from itertools import combinations


##############


def day1():
    data = [[-1 if line[0] == 'L' else 1, int(line[1:])] for line in open('input1.txt')]

    start_time = time.time()

    position = 50
    count_zero = 0
    for [direction, steps] in data:
        position = (position + steps * direction) % 100
        if position == 0:
            count_zero += 1


    position = 50
    hits, passes = 0, 0
    for [direction, steps] in data:
        if direction < 0:
            clicks_to_zero = position or 100
        else:
            clicks_to_zero = 100 - position

        position = (position + steps * direction) % 100
        if position == 0:
            hits += 1

        if steps >= clicks_to_zero:
            passes += (steps - clicks_to_zero) // 100 + 1


    task1 = hits
    task2 = passes

    return time.time() - start_time, task1, task2
    

##############

def find_invalid_ids(begin, end):
    invalid_ids = []

    if len(begin) % 2:  # uneven, start at 10..0, where len(10..0) == len(begin) + 1
        stock_id = "1" + "0" * len(begin)
    else:
        stock_id = begin
    rep_id = stock_id[:len(stock_id) // 2]

    r = range(int(begin), int(end) + 1)

    stop = False
    while not stop:
        double_id = int(rep_id * 2)
        if double_id in r:
            invalid_ids.append(double_id)
        rep_id = str(int(rep_id) + 1)

        if double_id >= r.stop:
            stop = True

    return invalid_ids

def find_rep_ids(start_block, begin_id, end_id):
    invalid_ids = []
    rep_id = start_block * 2
    while int(rep_id) <= end_id:
        if int(rep_id) >= begin_id:
            invalid_ids.append(int(rep_id))
        rep_id += start_block
    return invalid_ids


def find_repetitive_ids(begin, end):
    invalid_ids = []

    for i in range(1, int("1" + "0" * (len(end) // 2))) :
        start = str(i)
        invalid_ids.extend(find_rep_ids(start, int(begin), int(end)))


    return invalid_ids



def day2():
    data = [r.split('-') for r in open('input2.txt').readline().strip().split(',')]

    start_time = time.time()

    invalid_ids = [find_invalid_ids(r[0], r[1]) for r in data]

    task1 = sum(itertools.chain.from_iterable(invalid_ids))

    invalid_ids = [find_repetitive_ids(r[0], r[1]) for r in data]
    task2 = sum(set([x for r in invalid_ids for x in r]))

    return time.time() - start_time, task1, task2
    

##############

def findHighestJoltageOld(bank):
    joltage1 = "1"
    joltage2 = bank[-1]
    for j in reversed(bank[:-1]):
        if j >= joltage1:
            joltage2 = max(joltage1, joltage2)
            joltage1 = j

    return joltage1 + joltage2

def findHighestJoltageChain(bank, n = 12):
    joltage = list(bank[len(bank) - n:])
    indices = list(range(len(bank) - n, len(bank)))

    for i in reversed(range(0, len(bank) - n + 1)):
        if bank[i] >= joltage[0]:
            joltage[0] = bank[i]
            indices[0] = i

    for i in range(1, n):
        if indices[i - 1] + 1 >= indices[i]:
            break
        for j in reversed(range(indices[i - 1] + 1, len(bank) - n + 1 + i)):
            if bank[j] >= joltage[i]:
                joltage[i] = bank[j]
                indices[i] = j

    return int("".join(joltage))

def findHighestJoltage(bank):
    return findHighestJoltageChain(bank, 2)


def day3():
    banks = [line.strip() for line in open('input3.txt')]

    start_time = time.time()

    joltages = [findHighestJoltage(bank) for bank in banks]
    task1 = sum(joltages)

    joltages = [findHighestJoltageChain(bank) for bank in banks]
    task2 = sum(joltages)

    return time.time() - start_time, task1, task2
    

##############

def neighbour_roll_count(r, c, roll_map):
    directions = (-1, 0, 1)
    neighbour_count = 0

    for rdir in directions:
        rnbour = r + rdir
        if rnbour in range(len(roll_map)):
            for cdir in directions:
                cnbour = c + cdir
                if cnbour in range(len(roll_map[0])) and roll_map[rnbour][cnbour] == '@':
                    neighbour_count += 1

    return neighbour_count - 1 # remove the roll itself


def forkliftable(roll_map):
    roll_count = 0
    for r in range(len(roll_map)):
        for c in range(len(roll_map[0])):
            if roll_map[r][c] == '@':
                if neighbour_roll_count(r, c, roll_map) < 4:
                    roll_count += 1

    return roll_count

def neighbours(r, c, roll_map):
    directions = (-1, 0, 1)
    neighbours = set()

    for rdir in directions:
        rnbour = r + rdir
        if rnbour in range(len(roll_map)):
            for cdir in directions:
                cnbour = c + cdir
                if cnbour in range(len(roll_map[0])) and roll_map[rnbour][cnbour] == '@':
                    neighbours.add((rnbour, cnbour))

    return neighbours # includes the roll itself: (r,c)

def map_rolls(roll_map):
    rolls = {}

    for r in range(len(roll_map)):
        for c in range(len(roll_map[0])):
            if roll_map[r][c] == '@':
                rolls[(r, c)] = neighbours(r, c, roll_map)

    return rolls

def forklift(rolls):
    liftable = []
    for roll in rolls:
        if len(rolls[roll]) < 5:
            for n in rolls[roll]:
                if n != roll:
                    rolls[n].discard(roll)
            liftable.append(roll)

    for r in liftable:
        del rolls[r]

    return len(liftable)




def day4():
    data = [line.strip() for line in open('input4.txt')]
    start_time = time.time()

    task1 = forkliftable(data)

    rolls = map_rolls(data)
    total_lifted = 0
    lifted = 1
    while lifted > 0:
        lifted = forklift(rolls)
        total_lifted += lifted

    task2 = total_lifted

    return time.time() - start_time, task1, task2
    

##############

def get_ingredient_ranges(inventory):
    ranges = []
    for line in inventory:
        if not line:
            break
        start, stop = line.split('-')
        ranges.append(range(int(start), int(stop) + 1))

    return ranges

def merge_ranges(ranges):
    fresh_dict = {r.start : r.stop - 1 for r in ranges}
    starts = sorted(list(fresh_dict.keys()))

    mega_ranges = [(starts[0], fresh_dict[starts[0]])]
    for start in starts[1:]:
        if start <= mega_ranges[-1][1]:
            ids = mega_ranges.pop(-1)
            mega_ranges.append((ids[0], max(fresh_dict[start], ids[1])))
        else:
            mega_ranges.append((start, fresh_dict[start]))

    count_ids = 0
    for r in mega_ranges:
        count_ids += r[1] - r[0] + 1
    return count_ids

def merge_ranges2(ranges):
    ranges.sort(key = lambda r: (r.start, r.stop))
    merged = [ranges[0]]
    for r in ranges[1:]:
        if r.start <= merged[-1].stop:
            old = merged.pop(-1)
            merged.append(range(old.start, max(old.stop, r.stop)))
        else:
            merged.append(r)
    return sum([len(r) for r in merged])

def day5():
    data = [line.strip() for line in open('input5.txt')]

    inventory_ranges = get_ingredient_ranges(data)
    ingredients = [int(x) for x in data[len(inventory_ranges) + 1:]]

    start_time = time.time()

    count_fresh = 0
    for ingredient in ingredients:
        for fresh_ids in inventory_ranges:
            if ingredient in fresh_ids:
                count_fresh += 1
                break

    task1 = count_fresh

    task2 = merge_ranges2(inventory_ranges)

    return time.time() - start_time, task1, task2
    



##############

def octo_maths(worksheet, ops):
    results = []
    for i in range(len(ops)):
        results.append(list(itertools.accumulate(worksheet[i], operator.add if ops[i] == '+' else operator.mul))[-1])

    return results

def r2l_octo_maths(numbers, ops):
    r2l_worksheet = []

    arguments = []
    for c in range(len(numbers[0])):
        num = []
        for row in numbers:
            if row[c] != ' ':
                num.append(row[c])

        if num:
            arguments.append(''.join(num))
        else:
            r2l_worksheet.append(arguments)
            arguments = []

    r2l_worksheet.append(arguments)
    r2l_worksheet = str2int(r2l_worksheet)
    return octo_maths(r2l_worksheet, ops)

def str2int(array):
    return [[int(x) for x in row] for row in array]


def day6():
    data = [line.strip() for line in open('input6.txt')]
    data1 = [line.split() for line in data]
    numbers = str2int(data1[:-1])
    ops = data1[-1]

    start_time = time.time()

    num_array = np.array(numbers)
    task1 = sum(octo_maths(num_array.T, ops))
    task2 = sum(r2l_octo_maths(data[:-1], ops))

    return time.time() - start_time, task1, task2
    


##############

def dimensions(canvas):
    return len(canvas), len(canvas[0])

def split_beam(row, col, no_cols):
    beams = set()
    if col - 1 >= 0:
        beams.add(col - 1)

    if col + 1 < no_cols:
        beams.add(col + 1)

    return beams

def paint_by_numbers(canvas):
    rows, cols = dimensions(canvas)

    beams = {i: set() for i in range(rows)}
    beams[0] = {canvas[0].index('S')}
    splitters = set()

    for row in range(rows - 1):
        for col in beams[row]:
            if canvas[row][col] == '^':
                splitters.add((row, col))
                beams[row + 1].update(split_beam(row + 1, col, cols))
            else:
                beams[row + 1].add(col)

    return beams, splitters

def quantum_paint_by_numbers(canvas):
    rows, cols = dimensions(canvas)

    beams = {i: dict() for i in range(rows)}
    beams[0] = {canvas[0].index('S'): 1}

    for row in range(rows - 1):
        for col in beams[row]:
            new_cols = split_beam(row + 1, col, cols) if canvas[row][col] == '^' else [col]

            for c in new_cols:
                if c not in beams[row + 1]:
                    beams[row + 1][c] = 0
                beams[row + 1][c] += beams[row][col]

    return beams


def day7():
    canvas = [line.strip() for line in open('input7.txt')]
    start_time = time.time()

    beams, splitters = paint_by_numbers(canvas)
    task1 = len(splitters)

    beams = quantum_paint_by_numbers(canvas)
    task2 = sum(beams[len(beams) - 1].values())

    return time.time() - start_time, task1, task2
    

##############

class Distance:
    def __init__(self, p, q):
        self.p = p
        self.q = q
        self.distance = math.dist(p, q)

    def __repr__(self):
        return f"Distance({self.p}, {self.q}, {self.distance})"

# 162,817,812 and 425,690,689

def all_pairwise_distances(points):
    distances = []
    for p in range(len(points) - 1):
        for q in range(p + 1, len(points)):
            distances.append(Distance(points[p], points[q]))
    return sorted(distances, key=lambda x: x.distance)

def find_shortest_clusters(distances, n = 10):
    clusters = []
    for i in range(n):
        found_cluster = None
        delete_clusters = []

        for c in clusters:
            if distances[i].p in c or distances[i].q in c:
                c.add(distances[i].p)
                c.add(distances[i].q)

                if found_cluster is None:
                    found_cluster = c
                else:
                    found_cluster.update(c)
                    delete_clusters.append(c)

        if found_cluster is None:
            clusters.append({distances[i].p, distances[i].q})
        else:
            for c in delete_clusters:
                clusters.remove(c)

    return clusters

def find_single_cluster(distances, no_junctions):
    clusters = []
    i = 0
    while i < len(distances):
        found_cluster = None
        delete_clusters = []

        for c in clusters:
            if distances[i].p in c or distances[i].q in c:
                c.add(distances[i].p)
                c.add(distances[i].q)

                if found_cluster is None:
                    found_cluster = c
                else:
                    found_cluster.update(c)
                    delete_clusters.append(c)

        if found_cluster is None:
            clusters.append({distances[i].p, distances[i].q})
        else:
            for c in delete_clusters:
                clusters.remove(c)

            if len(found_cluster) == no_junctions:
                break

        i += 1

    if len(clusters) != 1 or len(clusters[0]) != no_junctions:
        raise Exception("No single cluster found")

    return distances[i].p[0] * distances[i].q[0]



def day8():
    data = [line.strip() for line in open('input8.txt')]
    start_time = time.time()

    points = [tuple([int(x) for x in line.split(',')]) for line in data]
    distances = all_pairwise_distances(points)
    clusters = find_shortest_clusters(distances, 1000)
    cluster_sizes = sorted([len(c) for c in clusters], reverse=True)

    task1 = functools.reduce(operator.mul, cluster_sizes[0:3])
    task2 = find_single_cluster(distances, len(points))

    return time.time() - start_time, task1, task2
    

##############

def squarea(tiles):
    pairs = itertools.combinations(tiles, 2)
    return [abs(p[0] - q[0] + 1) * abs(p[1] - q[1] + 1) for p, q in pairs]

def tile_loop(tiles):
    loop = {tiles[0]: [tiles[-1]]}

    for i in range(tiles[:-1]):
        loop[tiles[i]].append(tiles[i + 1])
        loop[tiles[i+1]] = [tiles[i]]

    loop[tiles[-1]].append(tiles[0])

    return loop



def day9():
    data = [line.strip() for line in open('input9.txt')]
    tiles = [tuple([int(x) for x in tile.split(',')]) for tile in data]

    start_time = time.time()

    task1 = max(squarea(tiles))
    task2 = None

    return time.time() - start_time, task1, task2
    

##############

def parse_buttons(button_schematic):
#    print(button_schematic)

    buttons = []

    for button in button_schematic:
#        print(button[1:-1].split(','))
        b = []
        for x in str(button[1:-1]).split(','):
            b.append(int(x))
        buttons.append(b)

    return buttons

#    print(button_schematic[0][1:-1].split(','))
#    return [tuple([int(x)]) for button in button_schematic for x in str(button[1:-1]).split(',')]


def parse_factory_machines(schematics):
    # input: [['[.##.]', '(3)', '(1,3)', '(2)', '(2,3)', '(0,2)', '(0,1)', '{3,5,4,7}'], ...]
    # output: [['.##.', [(3), (1,3), (2), (2,3), (0,2), (0,1)], '{3,5,4,7}'], ...]

    machines = []

    for machine in schematics:
        machines.append([machine[0][1:-1], parse_buttons(machine[1:-1]), machine[-1]])

    # ['.##.', [(3) (1,3) (2) (2,3) (0,2) (0,1)], '{3,5,4,7}']
    return machines


def day10():
    data = [line.strip() for line in open('input10.txt')]
    start_time = time.time()

    task1 = None
    task2 = None

    return time.time() - start_time, task1, task2
    

##############

def build_device_graph(devices):
    # aaa: you hhh
    graph = {}
    for device in devices:
        graph[device[0][:-1]] = device[1:]

    return graph

def reverse_device_graph(devices):
    # aaa: you hhh
    graph = {}
    for device in devices:
        for output in device[1:]:
            if output in graph:
                graph[output].append(device[0][:-1])
            else:
                graph[output] = [device[0][:-1]]

    return graph

class Paths:
    def __init__(self, graph):
        self.graph = graph

    @functools.cache
    def count_paths(self, node):
        return 1 if node == "out" else sum([self.count_paths(n) for n in self.graph[node]])

    @functools.cache
    def count_dac_fft_paths(self, node, dac = False, fft = False):
        match node:
            case "out": return 1 if (dac and fft) else 0
            case "dac": dac = True
            case "fft": fft = True

        return sum([self.count_dac_fft_paths(n, dac, fft) for n in self.graph[node]])

def day11():
    data = [line.strip().split(' ') for line in open('input11.txt')]
    start_time = time.time()

    graph = build_device_graph(data)

    paths = Paths(graph)
    task1 = paths.count_paths("you")
    task2 = paths.count_dac_fft_paths("svr")

    return time.time() - start_time, task1, task2
    

##############

"""Naive solution(s) for task 1: 
* (very naive solution) each gift is size <= 7. If the containing grid is < 7 * no. of gifts - answer is no 
* check if bounding boxes fit in the given grid - if so, all good.
(Thanks subreddit :-)
"""

def enough_spaces(no_gifts, xmas_tree_space):
    gift_space = 7 * no_gifts
    return gift_space <= xmas_tree_space[0] * xmas_tree_space[1]

def bounding_boxes_fit(no_gifts, xmas_tree_space):
    return xmas_tree_space[0] >= no_gifts * 3 and xmas_tree_space[1] >= no_gifts * 3

def day12():
    data = [line.strip() for line in open('input12.txt')]

    trees = [tree.split(':') for tree in data[31:]]

    tree_spaces = [[int(d) for d in tree[0].split('x')] for tree in trees]
    gifts = [[int(gift) for gift in tree[-1].strip().split(' ')] for tree in trees]

    start_time = time.time()

    task1 = sum([enough_spaces(sum(gifts[i]), tree_spaces[i]) for i in range(len(trees))])
    task2 = None

    return time.time() - start_time, task1, task2
    