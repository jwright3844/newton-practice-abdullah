import pytest
import numpy as np
import newton


def test_optimize():
    """
    Tests our optimize function with three test cases.
    """

    def f_1(x):
        return (x - 3) ** 2

    def f_2(x):
        return np.sqrt(x**2 + 3)

    def f_3(x):
        return x**4 / 4 - x**3 - x

    case_threshold = 0.001
    print("Test Case 1:")
    assert np.isclose(newton.optimize(1, f_1), 3)
    print("Success")
    print("Test Case 2:")
    print(newton.optimize(1, f_2))
    assert np.isclose(newton.optimize(0.5, f_2), 0, atol=0.0001)
    print("Success")
    print("Test Case 3:")
    assert np.isclose(newton.optimize(2.95, np.cos), np.pi)
    print("Success")
    print(newton.optimize(3, f_3))
    with pytest.raises(ZeroDivisionError):
        newton.optimize(0, f_3)


def test_bad_input():
    with pytest.raises(TypeError, match="`x_0` must be numeric"):
        newton.optimize(np.cos, 2.95)

    match_err_str = f"Argument is not a function, it is of type <class 'float'>"
    with pytest.raises(TypeError, match=match_err_str):
        newton.optimize(2.95, 2.95)
