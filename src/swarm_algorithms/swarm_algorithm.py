from src.callbacks import CallbackContainer

import numpy as np


class SwarmIntelligence:
    def __init__(self, population_size, nb_features, constraints, seed=None):
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
        self._step_number = None

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

    def populate_swarm(self, spawn_boundaries):
        spawn_boundaries = np.array(spawn_boundaries)
        minimums = spawn_boundaries[:, 0]
        maxes = spawn_boundaries[:, 1]
        population = self._rng.uniform(minimums, maxes, size=(self.population_size, self.nb_features))
        if self.constraints:
            for i in range(self.population_size):
                while not self.constraints.check(population[i, :].reshape(1, -1)):
                    population[i, :] = self._rng.uniform(minimums, maxes, size=(1, self.nb_features))
        self.population = population

        fit_values = self.calculate_fitness(population)

        self.local_best_solutions = population
        self.global_best_solution = population[np.argmin(fit_values[:, 0])]
    
        self.current_local_fitness = fit_values
        self.current_global_fitness = np.min(fit_values[:, 0])

        return population

    def update_best_local_global(self, population):
        current_fitness = self.calculate_fitness(population)
        mask = current_fitness < self.current_local_fitness

        self.local_best_solutions = mask * population + (1 - mask) * self.local_best_solutions
        self.current_local_fitness = mask * current_fitness + (1 - mask) * self.current_local_fitness

        best_solution_fitness = np.min(current_fitness[:, 0])

        if best_solution_fitness < self.current_global_fitness:
            self.current_global_fitness = best_solution_fitness
            self.global_best_solution = population[np.argmin(current_fitness[:, 0])]

    def calculate_fitness(self, population):
        fitness = self.fit_function(population).reshape(self.population_size, 1)
        mask = ~self.constraints.check(population)
        fitness[mask, :] = 1e15  # np.inf
        return fitness

    def get_new_positions(self, step_number):
        raise NotImplementedError

    def update_positions(self, new_positions, step):
        self.population = new_positions
        return self.population

    def go_swarm_go(self, stop_condition, callbacks=None):
        assert self.is_compiled, "Algorithm is not compiled, use `compile` method"
        self._step_number = 0
        all_callbacks = CallbackContainer(callbacks)
        all_callbacks.initialize_callback(self)

        all_callbacks.on_optimization_start()
        while not stop_condition.check(self):
            all_callbacks.on_epoch_start()
            pos = self.get_new_positions(self._step_number)
            self.update_positions(pos, self._step_number)
            self.update_best_local_global(self.population)
            self._step_number += 1
            all_callbacks.on_epoch_end()

        self._step_number = None
        all_callbacks.on_optimization_end()

        return self.global_best_solution

    @property
    def is_compiled(self):
        return self._compiled
