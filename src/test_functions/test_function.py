class Function:

    @property
    def minimal_value(self):
        raise NotImplementedError

    def fitness_function(self, x):
        raise NotImplementedError

    def optimal_solution(self, nb_dimensions):
        raise NotImplementedError