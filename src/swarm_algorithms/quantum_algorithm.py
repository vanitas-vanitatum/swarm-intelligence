from src.swarm_algorithms.swarm_algorithm import SwarmIntelligence

import numpy as np


class QuantumDeltaParticleSwarmOptimization(SwarmIntelligence):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1330875

    def __init__(self, population_size, nb_features, constraints, delta_potential_length_parameter, seed=0xCAFFE):
        super().__init__(population_size, nb_features, constraints, seed)
        self.g = delta_potential_length_parameter
        self.population_velocities = None

        self.global_best_solution = None
        self.local_best_solution = None

        self.current_global_fitness = None
        self.current_local_fitness = None

    def populate_swarm(self, search_space_boundaries):
        population = super().populate_swarm(search_space_boundaries)
        fit_values = self.fit_function(population)

        self.local_best_solution = population
        self.global_best_solution = population[np.argmax(fit_values[:, 0])]

        self.current_local_fitness = fit_values
        self.current_global_fitness = np.max(fit_values[:, 0])

        return population

    def go_swarm_go(self):
        pass

    def update_positions(self, new_positions, step):
        self.population = new_positions
        self._update_best_local_global(self.population)
        return self.population

    def _update_best_local_global(self, population):
        current_fitness = self.fit_function(population)
        mask = current_fitness > self.current_local_fitness

        self.local_best_solution = mask * population + (1 - mask) * self.local_best_solution
        self.current_local_fitness = mask * current_fitness + (1 - mask) * self.current_local_fitness

        best_solution_fitness = np.max(current_fitness[:, 0])

        if best_solution_fitness > self.current_global_fitness:
            self.current_global_fitness = best_solution_fitness
            self.global_best_solution = population[np.argmax(current_fitness[:, 0])]

    def get_new_positions(self, step_number):
        phi_1 = self._rng.rand(self.population_size, self.nb_features)
        phi_2 = self._rng.rand(self.population_size, self.nb_features)

        p = (phi_1 * self.local_best_solution + phi_2 * self.global_best_solution) / (phi_1 + phi_2)
        u = self._rng.rand(self.population_size, self.nb_features)

        L = (1 / self.g) * np.abs(self.population - p)
        mask = self._rng.rand(self.population_size, self.nb_features) > 0.5
        new_positions = mask * (p - L * np.log(1 / u)) + (1 - mask) * (p + L * np.log(1 / u))
        return new_positions
