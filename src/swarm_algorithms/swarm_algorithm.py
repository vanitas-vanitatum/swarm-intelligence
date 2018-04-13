import numpy as np


class SwarmIntelligence:
    def __init__(self, population_size, nb_features, constraints, seed=0xCAFFE):
        self.population_size = population_size
        self.nb_features = nb_features
        self.global_best_solution = None
        self.local_best_solutions = None
        self.population = None
        self.constraints = constraints
        self.fit_function = None
        self._compiled = False

        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)

    def compile(self, fit_function, search_space_boundaries):
        """
        `Compiles` model so it will use particular fitness function and populates swarm.
        :param fit_function: Fit function to use
        :param search_space_boundaries:
            Search space for each dimension. It should be in form of list of tuples: [(b_{min}, b_{max}), ...],
            where num of tuples is equal to number of dimensions
        :return: None
        """
        self.fit_function = fit_function
        self.populate_swarm(search_space_boundaries)
        self._compiled = True

    def populate_swarm(self, search_space_boundaries):
        search_space_boundaries = np.array(search_space_boundaries)
        minimums = search_space_boundaries[:, 0]
        maxes = search_space_boundaries[:, 1]
        population = self._rng.uniform(minimums, maxes, size=(self.population_size, self.nb_features))
        self.population = population
        return population

    def get_new_positions(self, step_number):
        raise NotImplementedError

    def update_positions(self, new_positions, step):
        raise NotImplementedError

    def go_swarm_go(self):
        raise NotImplementedError

    @property
    def is_compiled(self):
        return self._compiled
