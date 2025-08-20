import numpy as np


def f_d(x, f, epsilon):
    """
    First Derivative Finite Difference Function

    Parameters:
    --------------
    x: Point on function where we want to approximate first derivative
    f: Function
    epsilon: Difference used to approximate finite difference
    """
    return (f(x + epsilon) - f(x)) / epsilon


def f_dd(x, f, epsilon):
    """
    Second Derivative Finite Difference Function

    Parameters:
    --------------
    x: Point on function where we want to approximate first derivative
    f: Function
    epsilon: Difference used to approximate finite difference
    """
    return (f(x + epsilon) - 2 * f(x) + f(x - epsilon)) / (epsilon**2)


def optimize(x_0, f, epsilon=0.0001, threshold=0.0001):
    """
    Runs Newton's method to approximate maxima/minima of a function

    Parameters:
    --------------
    x: Starting point of Newton's method
    f: Function we want to find the maxima/minima of
    epsilon: Difference used to calculate finite difference derivatives
    threshold: How close we want to get to minima/maxima before return
    """
    curr_x = x_0
    prev_x = np.inf
    while np.abs(curr_x - prev_x) > threshold:
        prev_x = curr_x
        curr_x -= f_d(curr_x, f, epsilon) / f_dd(curr_x, f, epsilon)
    return curr_x


def test_optimize():
    """
    Tests our optimize function with three test cases.
    """

    def f_1(x):
        return (x - 3) ** 2

    def f_2(x):
        return np.sqrt(x**2 + 3)

    case_threshold = 0.001
    print("Test Case 1:")
    assert np.abs(optimize(1, f_1) - 3) < case_threshold
    print("Success")
    print("Test Case 2:")
    assert np.abs(optimize(1, f_2) - 0) < case_threshold
    print("Success")
    print("Test Case 3:")
    assert np.abs(optimize(3, np.cos) - np.pi) < case_threshold
    print("Success")
