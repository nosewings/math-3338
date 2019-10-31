from scipy.optimize import linprog


def example_117():
    f = [1, 0, 0]
    coefficients = [
        [-1, 1, 0],
        [-1, -1, 0],
        [-1, 0, 1],
        [-1, 0, -1],
        [-1, 1, 1],
        [-1, -1, -1],
    ]
    bounds = [
        13,
        -13,
        7,
        -7,
        19,
        -19,
    ]
    return linprog(f, A_ub=coefficients, b_ub=bounds, method='revised simplex')
