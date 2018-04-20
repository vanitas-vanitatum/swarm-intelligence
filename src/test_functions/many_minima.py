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

    def minimal_value(self, nb_dimensions):
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

    def minimal_value(self, nb_dimensions):
        return 0.0

    def optimal_solution(self, nb_dimensions):
        return np.ones(1, nb_dimensions) * 420.9687

    def fitness_function(self, x, axis=1):
        x = np.asarray(x)
        param = 418.9829
        sqrt_abs = np.sqrt(np.abs(x))
        sum_sin_sqrt = (x * np.sin(sqrt_abs)).sum(axis=axis)
        return param * x.shape[axis] - sum_sin_sqrt


class Griewank(Function):
    """
    https://www.sfu.ca/~ssurjano/griewank.html
    """

    def minimal_value(self, nb_dimensions):
        return 0.0

    def fitness_function(self, x, axis=1):
        x = np.asarray(x)
        dim = x.shape[axis]
        sumterm = (np.square(x)/4000.).sum(axis=axis)
        i = np.arange(1, dim+1).reshape(1, -1)
        prodterm = np.cos(x/np.sqrt(i)).prod(axis=axis)

        return sumterm - prodterm + 1

    def optimal_solution(self, nb_dimensions):
        return np.zeros(1, nb_dimensions)


class Levy(Function):
    """
    https://www.sfu.ca/~ssurjano/levy.html
    """
    def minimal_value(self, nb_dimensions):
        return 0.0

    def fitness_function(self, x, axis=1):
        x = np.asarray(x)
        w = 1 + ((x - 1) / 4.)

        term1 = np.square(np.sin(np.pi * w[:, 0]))

        sumterm1 = np.square(w[:, :-1] - 1)
        sumterm2 = 1 + 10 * np.square(np.sin(np.pi * w[:, :-1] + 1))
        sumterm = (sumterm1 * sumterm2).sum(axis=axis)

        term3 = np.square(w[:, -1] - 1)
        term4 = 1 + np.square(np.sin(2 * np.pi * w[:, -1]))

        return term1 + sumterm + (term3 * term4)


    def optimal_solution(self, nb_dimensions):
        return np.ones(1, nb_dimensions)