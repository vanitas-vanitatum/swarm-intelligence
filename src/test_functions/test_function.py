class Function:

    def __init__(self):  # , constraints):
        pass
        # self.constraints = constraints

    def minimal_value(self, nb_dimensions):
        raise NotImplementedError

    def fitness_function_implementation(self, x, axis=1):
        raise NotImplementedError

    def fitness_function(self, x, axis=1):
        # mask = ~self.constraints.check(x)
        fitness = self.fitness_function_implementation(x, axis).reshape(x.shape[0], -1)
        # fitness[mask, :] = 1e15#np.inf
        return fitness

    def optimal_solution(self, nb_dimensions):
        raise NotImplementedError
