import numpy as np


def divided_differences(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y):
        raise ValueError
    n = len(x)
    ret = []
    acc = y
    for i in range(n-1):
        num = np.diff(acc)
        denom = x[i+1:] - x[:-(i+1)]
        acc = num / denom
        ret.append(acc)
    return ret


def recommend_order(x, y, strategy):
    table = divided_differences(x, y)
    for i, col in enumerate(table):
        if strategy(col):
            return i
    raise ValueError('no good order according to the given strategy')
