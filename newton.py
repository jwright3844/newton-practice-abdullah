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