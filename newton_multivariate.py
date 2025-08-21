import numdifftools as nd
import numpy as np
import warnings

def gradient(x, f):
    """
    First Derivative Finite Difference Function

    Parameters:
    --------------
    x: Point on function where we want to approximate first derivative
    f: Multivariate function (callable)
    """
    grad_fun_f = nd.Gradient(f)
    return np.linalg.matrix_transpose(grad_fun_f(x))


def hessian(x, f):
    """
    Second Derivative Finite Difference Function

    Parameters:
    --------------
    x: Point on function where we want to approximate first derivative
    f: Multivariate function (callable)
    """
    Hfun_f = nd.Hessian(f)
    return Hfun_f(x)


def optimize(x_0, f, threshold=1e-5):
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
    if not (isinstance(x_0, list) or isinstance(x_0, np.ndarray)):
        raise TypeError("`x_0` must be a list")
    if not callable(f):
        raise TypeError(f"Argument is not a function, it is of type {type(f)}")

    # Optimization
    curr_x = x_0
    prev_x = np.inf*np.ones_like(x_0)
    iter = 0
    while np.linalg.norm(curr_x - prev_x) > threshold:
        if iter > 10000:
            raise RuntimeError(f"At step {iter}, optimization does not converge")
        iter += 1
        prev_x = curr_x
        print("curr_x")
        print(curr_x)
        grad_f_x = gradient(curr_x, f)
        print("grad_f_x")
        print(grad_f_x)
        hessian_f_x = hessian(curr_x, f)
        print("hessian_f_x")
        print(hessian_f_x)
        # if np.isclose(f_dd_x, 0):
        #     raise ZeroDivisionError(
        #         f"Second Derivative is approximately 0 on step {iter}."
        #     )
        curr_x = curr_x - np.linalg.inv(hessian_f_x)*grad_f_x
    return curr_x

def f_1(x):
    return np.sum(x**2)
print(f_1(np.array([[1],[1],[1]])))
print(optimize(np.array([[1],[1],[1]]), f_1))
