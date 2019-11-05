import numpy as np
import scipy.linalg


def _spline_banded_matrix(h):
    h = np.asarray(h)
    n = len(h)
    ret = np.zeros([3, n+1])
    ret[1, 0] = 1
    ret[1, -1] = 1
    h0 = h[:n-1]
    h1 = h[1:n]
    ret[0, 1:n] = h0
    ret[1, 1:n] = 2.0*(h0 + h1)
    ret[2, 1:n] = h1
    return ret


def _spline_vector(a, h):
    a = np.asarray(a)
    h = np.asarray(h)
    n = len(h)
    ret = np.empty([n+1])
    ret[0] = 0
    ret[-1] = 0
    h0 = h[:-1]
    h1 = h[1:]
    a0 = a[:-2]
    a1 = a[1:-1]
    a2 = a[2:]
    ret[1:-1] = 3.0*((a2 - a1)/h1 - (a1 - a0)/h0)
    return ret


def _spline_coefficients(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    a = y
    h = np.diff(x)
    A = _spline_banded_matrix(h)
    v = _spline_vector(a, h)
    c = scipy.linalg.solve_banded([1, 1], A, v)
    b = np.diff(a)/h - (h*(c[1:] + 2.0*c[:-1]))/3.0
    d = np.diff(c)/(3.0*h)
    return a[:-1], b, c[:-1], d


def _untranslate_coefficients(x, a, b, c, d):
    x = x[:-1]
    x2 = x**2.0
    x3 = x**3.0
    a_ = a - b*x + c*x2 - d*x3
    b_ = b - 2.0*c*x + 3.0*d*x2
    c_ = c - 3.0*d*x
    d_ = d
    return a_, b_, c_, d_


def _spline_polys(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    a, b, c, d = _spline_coefficients(x, y)
    a, b, c, d = _untranslate_coefficients(x, a, b, c, d)
    return np.column_stack([d, c, b, a])


def coefficients(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if not np.all(np.diff(xs) > 0):
        raise ValueError
    polys = _spline_polys(xs, ys)
    return polys


class CubicSpline:
    def __init__(self, x, y):
        c = coefficients(x, y)
        self._start = x[0]
        self._end = x[-1]
        self._xs = x[:-1]
        self._c = c
        self._g = np.vectorize(self._f)

    @property
    def c(self):
        return self._c.T

    def __call__(self, x):
        return self._g(x)

    def _f(self, x):
        if x < self._start or x > self._end:
            return np.nan
        ixs, = np.where(x >= self._xs)
        ix = ixs[-1]
        return np.polyval(self._c[ix], x)
