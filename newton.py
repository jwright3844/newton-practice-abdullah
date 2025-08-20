import numpy as np

def f_d(x, f, epsilon):
    return (f(x + epsilon) - f(x))/epsilon

def f_dd(x, f, epsilon):
    return (f(x + epsilon) - 2*f(x) + f(x - epsilon))/(epsilon**2)

def optimize(x_0, f, epsilon=0.0001, threshold=0.0001):
    curr_x = x_0
    prev_x = np.inf
    while np.abs(curr_x -prev_x) > threshold:
        prev_x = curr_x
        curr_x -= f_d(curr_x, f, epsilon)/f_dd(curr_x, f, epsilon)
    return(curr_x)

def test_optimize():
    def f_1(x):
        return (x-3)**2
    def f_2(x):
        return np.sqrt(x**2 + 3)
    def f_3(x):
        return np.sin(x)

    case_threshold = 0.001
    print(f"Test Case 1:")
    assert np.abs(optimize(1, f_1) - 3) < case_threshold
    print(f"Success")
    print(f"Test Case 2:")
    assert np.abs(optimize(1, f_2) - 0) < case_threshold
    print(f"Success")
    print(f"Test Case 3:")
    assert np.abs(optimize(3, f_3) - np.pi) < case_threshold
    print(f"Success")