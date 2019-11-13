import numpy as np


def euler_forward_2d(f, x, y, h, n):
    xs = x + h*np.arange(n + 1)
    ys = np.empty([n + 1])
    ys[0] = y
    for (i, x) in enumerate(xs[:-1], 1):
        y += h*f(x, y)
        ys[i] = y
    return np.column_stack([xs, ys])


def euler_modified_2d(f, x, y, h, n):
    # Need one more point for the last correction step.
    xs = x + h*np.arange(n + 2)
    ys = np.empty([n + 1])
    ys[0] = y
    for (i, (xa, xb)) in enumerate(zip(xs[:-2], xs[1:-1]), 1):
        fxy = f(xa, y)
        y_ = y + h*fxy
        y = y + h*(fxy + f(xb, y_))/2.0
        ys[i] = y
    # Don't return the extra correction value.
    xs = xs[:-1]
    return np.column_stack([xs, ys])
