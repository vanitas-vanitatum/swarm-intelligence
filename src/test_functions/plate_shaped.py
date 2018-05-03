from src.test_functions.test_function import Function
import numpy as np


class Zakharov(Function):
    """
    https://www.sfu.ca/~ssurjano/zakharov.html
    """

    def minimal_value(self, nb_dimensions):
        return 0.0

    def fitness_function_implementation(self, x, axis=1):
        x = np.asarray(x)
        dim = x.shape[axis]
        i = np.arange(1, dim + 1)
        term1 = np.square(x).sum(axis=axis)
        term2 = (0.5 * i * x).sum(axis=axis)

        return term1 + np.square(term2) + np.power(term2, 4)

    def optimal_solution(self, nb_dimensions):
        return np.zeros((1, nb_dimensions))
