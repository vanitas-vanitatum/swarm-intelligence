from src.swarm_algorithms.swarm_algorithm import SwarmIntelligence
import numpy as np


class ParticleSwarmOptimisation(SwarmIntelligence):

    def __init__(self, population_size, nb_features, constraints, inertia=1, divergence=1, learning_factor_1=2, learning_factor_2=2):
        super().__init__(population_size, nb_features, constraints)
        self.population_velocities = None

        self.inertia = inertia
        self.divergence = divergence

        self.lf_1 = learning_factor_1
        self.lf_2 = learning_factor_2

    def go_swarm_go(self):
        pass

    def populate_swarm(self):
        # TODO: what values should be used? [0,1] or maybe [-inf, +inf]? or should constraints give min and max?
        pop = np.random.random((self.population_size, self.nb_features))
        raise NotImplementedError

    def update_positions(self, new_positions, step):
        pass

    def get_new_positions(self, step_number):
        v = self.calculate_velocities()
        return self.population + v

    def calculate_velocities(self):
        phi_1 = np.random.random((self.population_size, self.nb_features))
        phi_2 = np.random.random((self.population_size, self.nb_features))

        local_factor = self.lf_1 * phi_1 * (self.local_best_solutions - self.population)
        global_factor = self.lf_2 * phi_2 * (self.global_best_solution - self.population)
        v = self.inertia * self.population_velocities + local_factor + global_factor
        return self.divergence * v

