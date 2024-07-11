import random
import math


def fuzzy_ceil(fuzzy_num):
    a = math.ceil(fuzzy_num[0])
    b = math.ceil(fuzzy_num[1])
    c = math.ceil(fuzzy_num[2])
    ceil = (a, b, c)
    return ceil

def Fuzzified_Data(t):
    if t == float('inf') or t == 0:
        Fuzzy_time = (t, t, t)
        return Fuzzy_time
    delta1 = 0.75
    delta2 = 1.2  # verify that 𝛿_2 − 1 < 1 − 𝛿_1
    t_m = t
    t_l = round(random.uniform(delta1 * t_m, t_m), 4)
    t_u = round(random.uniform(t_m, min(delta2 * t_m, 2 * t_m - t_l)), 4)
    Fuzzy_time = (t_l, t_m, t_u)
    return Fuzzy_time


def fuzzy_mean(fuzzy_num):
    mean = (fuzzy_num[0] + 2 * fuzzy_num[1] + fuzzy_num[2]) / 4
    return mean


def fuzzy_std(fuzzy_num):
    var = (2 * (fuzzy_num[0] - fuzzy_num[1]) ** 2 +
           1 * (fuzzy_num[0] - fuzzy_num[2]) ** 2 +
           2 * (fuzzy_num[1] - fuzzy_num[2]) ** 2) / 80
    std = var ** 0.5
    return std


def fuzzy_fitness(fuzzy_num):
    eta = 1
    mean = fuzzy_mean(fuzzy_num)
    std = fuzzy_std(fuzzy_num)
    return mean + eta * std


def fuzzy_sum(tuple1, tuple2):
    t_l = tuple1[0] + tuple2[0]
    t_m = tuple1[1] + tuple2[1]
    t_u = tuple1[2] + tuple2[2]
    tuple3 = (t_l, t_m, t_u)
    return tuple3


def fuzzy_minus(tuple1, tuple2):
    t_l = tuple1[0] - tuple2[2]
    if t_l < 0.00:
        t_l = 0.00
    t_m = tuple1[1] - tuple2[1]
    t_u = tuple1[2] - tuple2[0]
    tuple3 = (t_l, t_m, t_u)
    return tuple3


def fuzzy_num_multiply(tuple1, num):
    t_l = tuple1[0] * num
    t_m = tuple1[1] * num
    t_u = tuple1[2] * num
    tuple2 = (t_l, t_m, t_u)
    return tuple2


def fuzzy_divide(tuple1, tuple2):
    if not isinstance(tuple1, tuple):
        tuple1 = (tuple1, tuple1, tuple1)
    if not isinstance(tuple2, tuple):
        tuple2 = (tuple2, tuple2, tuple2)
    t_l = tuple1[0] / tuple2[2]
    t_m = tuple1[1] / tuple2[1]
    t_u = tuple1[2] / tuple2[0]
    tuple3 = (t_l, t_m, t_u)
    return tuple3


def fuzzyToreal(tuple1):
    temp = (tuple1[2], tuple1[2], tuple1[2])
    return temp

def fuzzy_deadline(t):
    tmp = (t, t, t)
    return tmp

def fuzzy_rank(tuple1, tuple2):
    if fuzzy_mean(tuple1) > fuzzy_mean(tuple2):
        return True
    elif fuzzy_mean(tuple1) == fuzzy_mean(tuple2):
        if tuple1[1] > tuple2[1]:
            return True
        elif tuple1[1] == tuple2[1]:
            if tuple1[2] - tuple1[0] >= tuple2[2] - tuple2[0]:
                return True
    return False


def fuzzy_more_than(tuple1, tuple2):
    if tuple1[0] > tuple2[0] and tuple1[1] > tuple2[1] and tuple1[2] > tuple2[2]:
        return True
    return False



def fuzzy_less_than(tuple1, tuple2):
    if tuple1[0] <= tuple2[0] and tuple1[1] <= tuple2[1] and tuple1[2] <= tuple2[2]:
        return True
    return False


def fuzzy_max(tuple1, tuple2):
    """if fuzzy_rank(tuple1, tuple2):
        return tuple1
    else:
        return tuple2"""
    t_l = max(tuple1[0], tuple2[0])
    t_m = max(tuple1[1], tuple2[1])
    t_u = max(tuple1[2], tuple2[2])
    tuple3 = (t_l, t_m, t_u)
    return tuple3


def fuzzy_min(tuple1, tuple2):
    t_l = min(tuple1[0], tuple2[0])
    t_m = min(tuple1[1], tuple2[1])
    t_u = min(tuple1[2], tuple2[2])
    tuple3 = (t_l, t_m, t_u)
    return tuple3

def fuzzy_std(fuzzy_num):
    var = (2 * (fuzzy_num[0] - fuzzy_num[1]) ** 2 +
           1 * (fuzzy_num[0] - fuzzy_num[2]) ** 2 +
           2 * (fuzzy_num[1] - fuzzy_num[2]) ** 2) / 80
    std = var ** 0.5
    return std

def fuzzy_fitness(fuzzy_num):
    eta = 1
    mean = fuzzy_mean(fuzzy_num)
    std = fuzzy_std(fuzzy_num)
    return mean + eta * std