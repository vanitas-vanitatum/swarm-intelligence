import numpy as np
from src.test_functions.test_function import Function


class PermDBeta(Function):
    """
    https://www.sfu.ca/~ssurjano/perm0db.html
    """
    def __init__(self, beta=0.5):
        self.beta = beta

    def minimal_value(self, nb_dimensions):
        return 0

    def fitness_function(self, x, axis=1):
        x = np.asarray(x)
        dim = x.shape[axis]
        res = 0.
        for i in range(1, dim+1):
            outer = 0.
            for j in range(1, dim+1):
                inner = j**i + self.beta
                inner *= np.power((x[:, (j-1)] / j), i) - 1
                outer += inner
            res += outer ** 2
        return res

    def optimal_solution(self, nb_dimensions):
        return np.arange(1, nb_dimensions+1, 1).reshape(1, -1)


class StyblinskiTang(Function):
    """
    https://www.sfu.ca/~ssurjano/stybtang.html
    """

    def minimal_value(self, nb_dimensions):
        return -39.16599 * nb_dimensions

    def fitness_function(self, x, axis=1):
        x = np.asarray(x)
        term1 = np.power(x, 4)
        term2 = 16 * np.square(x)
        term3 = 5 * x
        return 0.5 * (term1 + term2 + term3).sum(axis=axis)

    def optimal_solution(self, nb_dimensions):
        return -2.903534 * np.ones(1, nb_dimensions)
