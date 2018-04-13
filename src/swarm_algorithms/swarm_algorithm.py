import numpy as np


class SwarmIntelligence:
    def __init__(self, population_size, nb_features, constraints, seed=0xCAFFE):
        self.population_size = population_size
        self.nb_features = nb_features

        self.global_best_solution = None
        self.local_best_solutions = None

        self.current_global_fitness = None
        self.current_local_fitness = None

        self.population = None
        self.constraints = constraints
        self.fit_function = None
        self._compiled = False

        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)

    def compile(self, fit_function, spawn_boundaries):
        """
        `Compiles` model so it will use particular fitness function and populates swarm.
        :param fit_function: Fit function to use
        :param spawn_boundaries:
            Search space for each dimension. It should be in form of list of tuples: [(b_{min}, b_{max}), ...],
            where num of tuples is equal to number of dimensions
        :return: None
        """
        self.fit_function = fit_function
        self.populate_swarm(spawn_boundaries)
        self._compiled = True

    def populate_swarm(self, search_space_boundaries):
        search_space_boundaries = np.array(search_space_boundaries)
        minimums = search_space_boundaries[:, 0]
        maxes = search_space_boundaries[:, 1]
        population = self._rng.uniform(minimums, maxes, size=(self.population_size, self.nb_features))
        for i in range(self.population_size):
            while not self.constraints.check(population[i, :]):
                population[i, :] = self._rng.uniform(minimums, maxes, size=(1, self.nb_features))
        self.population = population

        fit_values = self.fit_function(population)

        self.local_best_solutions = population
        self.global_best_solution = population[np.argmax(fit_values[:, 0])]

        self.current_local_fitness = fit_values
        self.current_global_fitness = np.max(fit_values[:, 0])

        return population

    def update_best_local_global(self, population):
        current_fitness = self.fit_function(population)
        mask = current_fitness > self.current_local_fitness

        self.local_best_solutions = mask * population + (1 - mask) * self.local_best_solutions
        self.current_local_fitness = mask * current_fitness + (1 - mask) * self.current_local_fitness

        best_solution_fitness = np.max(current_fitness[:, 0])

        if best_solution_fitness > self.current_global_fitness:
            self.current_global_fitness = best_solution_fitness
            self.global_best_solution = population[np.argmax(current_fitness[:, 0])]

    def get_new_positions(self, step_number):
        raise NotImplementedError

    def update_positions(self, new_positions, step):
        raise NotImplementedError

    def go_swarm_go(self):
        # TODO: To implement in separate task
        pass

    @property
    def is_compiled(self):
        return self._compiled
