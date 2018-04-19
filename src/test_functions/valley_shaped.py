import numpy as np
from src.test_functions.test_function import Function


class Rosenbrock(Function):
    """
    https://www.sfu.ca/~ssurjano/rosen.html
    """
    def minimal_value(self):
        return 0.0

    def fitness_function(self, x, axis=1):
        x = np.asarray(x)
        square_x = np.square(x)
        term1 = 100. * np.square(x[:, 1:] - square_x[:, :-1])
        square_x_1 = np.square(x-1)
        return (term1 + square_x_1[:, :-1]).sum(axis=axis)

    def optimal_solution(self, nb_dimensions):
        return np.ones(1, nb_dimensions)