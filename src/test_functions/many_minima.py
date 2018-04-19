import numpy as np
from src.test_functions.test_function import Function


class Ackley(Function):
    """
    https://www.sfu.ca/~ssurjano/ackley.html
    """
    def __init__(self, a=20, b=0.2, c=2*np.pi):
        self.a = a
        self.b = b
        self.c = c

    def minimal_value(self):
        return 0.0

    def optimal_solution(self, nb_dimensions):
        return np.zeros(1, nb_dimensions)

    def fitness_function(self, x, axis=1):
        x = np.asarray(x)
        factor1 = np.sqrt(1. * np.square(x).sum(axis=axis) / x.shape[axis])
        factor1 = -self.a * np.exp(-self.b * factor1)
        factor2 = np.exp(1. * np.cos(self.c * x).sum(axis=axis) / x.shape[axis])
        return factor1 - factor2 + self.a + np.e


class Shwefel(Function):
    """
    https://www.sfu.ca/~ssurjano/schwef.html
    """
    def __init__(self):
        pass

    def minimal_value(self):
        return 0.0

    def optimal_solution(self, nb_dimensions):
        return np.ones(1, nb_dimensions) * 420.9687

    def fitness_function(self, x, axis=1):
        x = np.asarray(x)
        param = 418.9829
        sqrt_abs = np.sqrt(np.abs(x))
        sum_sin_sqrt = (x * np.sin(sqrt_abs)).sum(axis=axis)
        return param * x.shape[axis] - sum_sin_sqrt
