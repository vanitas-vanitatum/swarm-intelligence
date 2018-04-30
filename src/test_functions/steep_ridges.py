import numpy as np
from src.test_functions.test_function import Function


class Michalewicz(Function):
    """
    https://www.sfu.ca/~ssurjano/michal.html
    """
    def __init__(self, m=10):
        self.m = m

    def minimal_value(self, nb_dimensions):
        """
        https://www.sfu.ca/~ssurjano/michal.html
        different with each dimensionality, not easy to determine
        :param nb_dimensions:
        """
        return None

    def optimal_solution(self, nb_dimensions):
        """
        https://www.sfu.ca/~ssurjano/michal.html
        different with each dimensionality, not easy to determine
        """
        if nb_dimensions==2:
            return np.asarray([[2.20, 1.57]])
        else:
            return None

    def fitness_function_implementation(self, x, axis=1):
        x = np.asarray(x)
        dim = x.shape[axis]
        i = np.arange(1, dim+1, 1)
        sin_2m = np.power(np.sin(i * np.square(x)/np.pi), 2*self.m)
        return - (np.sin(x) * sin_2m).sum(axis=axis)
