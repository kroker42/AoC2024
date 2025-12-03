import re
import time
import math
import operator
import itertools

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
    