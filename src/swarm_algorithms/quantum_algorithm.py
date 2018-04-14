from src.swarm_algorithms.swarm_algorithm import SwarmIntelligence

import numpy as np


class QuantumDeltaParticleSwarmOptimization(SwarmIntelligence):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1330875

    def __init__(self, population_size, nb_features, constraints, delta_potential_length_parameter, seed=None):
        super().__init__(population_size, nb_features, constraints, seed)
        self.g = delta_potential_length_parameter

    def get_new_positions(self, step_number):
        phi_1 = self._rng.rand(self.population_size, self.nb_features)
        phi_2 = self._rng.rand(self.population_size, self.nb_features)

        p = (phi_1 * self.local_best_solutions + phi_2 * self.global_best_solution) / (phi_1 + phi_2)
        u = self._rng.rand(self.population_size, self.nb_features)

        L = (1 / self.g) * np.abs(self.population - p)
        mask = self._rng.rand(self.population_size, self.nb_features) > 0.5
        new_positions = mask * (p - L * np.log(1 / u)) + (1 - mask) * (p + L * np.log(1 / u))
        return new_positions
