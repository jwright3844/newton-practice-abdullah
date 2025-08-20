import numpy as np
import warnings


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


def optimize(x_0, f, epsilon=1e-5, threshold=1e-5):
    """
    Runs Newton's method to approximate maxima/minima of a function

    Parameters:
    --------------
    x: Starting point of Newton's method
    f: Function we want to find the maxima/minima of
    epsilon: Difference used to calculate finite difference derivatives
    threshold: How close we want to get to minima/maxima before return
    """
    # Check Inputs
    if not (isinstance(x_0, float) or isinstance(x_0, int)):
        raise TypeError("`x_0` must be numeric")
    if not callable(f):
        raise TypeError(f"Argument is not a function, it is of type {type(f)}")

    # Optimization
    curr_x = x_0
    prev_x = np.inf
    iter = 0
    while np.abs(curr_x - prev_x) > threshold:
        if iter > 10000:
            raise RuntimeError(f"At step {iter}, optimization does not converge")
        iter += 1
        prev_x = curr_x
        f_d_x = f_d(curr_x, f, epsilon)
        f_dd_x = f_dd(curr_x, f, epsilon)
        if np.isclose(f_dd_x, 0):
            raise ZeroDivisionError(
                f"Second Derivative is approximately 0 on step {iter}."
            )
        curr_x = curr_x - f_d_x / f_dd_x
    return curr_x
