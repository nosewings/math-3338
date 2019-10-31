import numpy as np


def P(xs, ys, x):

    if len(xs) != len(ys):
        raise ValueError
    n = len(xs)

    def L(k):
        x_k = xs[k]
        xs_k = np.delete(xs, k)
        return (x - xs_k).prod() / (x_k - xs_k).prod()

    return (ys * [L(k) for k in range(n)]).sum()
