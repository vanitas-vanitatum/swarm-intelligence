class Function:

    def minimal_value(self, nb_dimensions):
        raise NotImplementedError

    def fitness_function(self, x, axis=1):
        raise NotImplementedError

    def optimal_solution(self, nb_dimensions):
        raise NotImplementedError