import numpy as np
import scipy.integrate


def euler(f, y0, t):
    # Turn `y0` into an array if it's a scalar. Why? So that we always return
    # the same shape as `odeint`.
    y0 = np.atleast_1d(y0)
    ys = np.empty([len(t), *np.shape(y0)])
    ys[0] = y0
    y = y0
    for (i, (t0, t1)) in enumerate(zip(t[:-1], t[1:]), 1):
        dt = t1 - t0
        y += dt*f(y, t0)
        ys[i] = y
    return ys


def heun(f, y0, t):
    y0 = np.atleast_1d(y0)
    ys = np.empty([len(t), *np.shape(y0)])
    ys[0] = y0
    y = y0
    for (i, (t0, t1)) in enumerate(zip(t[:-1], t[1:]), 1):
        dt = t1 - t0
        # Avoid computing this twice.
        f_y_t0 = f(y, t0)
        # Prediction.
        y_ = y + dt*f_y_t0
        # Correction.
        y += dt*(f_y_t0 + f(y_, t1))/2.0
        ys[i] = y
    return ys


def rk4(f, y0, t):
    y0 = np.atleast_1d(y0)
    ys = np.empty([len(t), *np.shape(y0)])
    ys[0] = y0
    y = y0
    for (i, (t0, t2)) in enumerate(zip(t[:-1], t[1:]), 1):
        dt = t2 - t0
        t1 = t0 + dt/2.0
        k1 = dt*f(y, t0)
        k2 = dt*f(y + k1/2.0, t1)
        k3 = dt*f(y + k2/2.0, t1)
        k4 = dt*f(y + k3, t2)
        y += (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
        ys[i] = y
    return ys
