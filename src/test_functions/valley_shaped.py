import numpy as np
from src.test_functions.test_function import Function


class Rosenbrock(Function):
    """
    https://www.sfu.ca/~ssurjano/rosen.html
    """
    def minimal_value(self, nb_dimensions):
        return 0.0

    def fitness_function(self, x, axis=1):
        x = np.asarray(x)
        square_x = np.square(x)
        term1 = 100. * np.square(x[:, 1:] - square_x[:, :-1])
        square_x_1 = np.square(x-1)
        return (term1 + square_x_1[:, :-1]).sum(axis=axis)

    def optimal_solution(self, nb_dimensions):
        return np.ones(1, nb_dimensions)


class DixonPrice(Function):
    """
    https://www.sfu.ca/~ssurjano/dixonpr.html
    """

    def minimal_value(self, nb_dimensions):
        return 0.0

    def fitness_function(self, x, axis=1):
        x = np.asarray(x)
        dim = x.shape[axis]
        term1 = np.square(x[:, 0] - 1)

        i = np.arange(2, dim+1)
        term2 = np.square(i * (2 * np.square(x[:, 1:]) - x[:, :-1]))
        term2 = term2.sum(axis=axis)
        return term1 + term2

    def optimal_solution(self, nb_dimensions):
        i = np.arange(1, nb_dimensions + 1)
        exp_i = np.power(2, i)
        exponent = - (exp_i - 2) / exp_i
        return np.power(2, exponent)