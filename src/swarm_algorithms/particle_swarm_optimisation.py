from src.swarm_algorithms.swarm_algorithm import SwarmIntelligence
import numpy as np


class ParticleSwarmOptimisation(SwarmIntelligence):

    def __init__(self, population_size, nb_features, constraints, inertia=1., divergence=1., learning_factor_1=2.,
                 learning_factor_2=2., seed=None):
        super().__init__(population_size, nb_features, constraints, seed)
        self.population_velocities = None

        self.inertia = inertia
        self.divergence = divergence

        self.lf_1 = learning_factor_1
        self.lf_2 = learning_factor_2

    def populate_swarm(self, spawn_boundaries):
        super().populate_swarm(spawn_boundaries)

        spawn_boundaries = np.array(spawn_boundaries)
        minimums = spawn_boundaries[:, 0]
        maxes = spawn_boundaries[:, 1]

        self.population_velocities = self._rng.uniform(-np.abs(maxes - minimums),
                                                       np.abs(maxes - minimums),
                                                       size=(self.population_size, self.nb_features))

    def get_new_positions(self, step_number):
        v = self.calculate_velocities(step_number)
        self._update_current_velocities(v)
        return self.population + v

    def _update_current_velocities(self, new_velocities):
        self.population_velocities = new_velocities

    def calculate_velocities(self, step):
        phi_1 = self._rng.rand(self.population_size, self.nb_features)
        phi_2 = self._rng.rand(self.population_size, self.nb_features)

        local_factor = self.lf_1 * phi_1 * (self.local_best_solutions - self.population)
        global_factor = self.lf_2 * phi_2 * (self.global_best_solution - self.population)
        v = self.inertia * self.population_velocities + local_factor + global_factor
        return (self.divergence ** step) * v


class BasicPSO(ParticleSwarmOptimisation):

    def __init__(self, population_size, nb_features, constraints,
                 learning_factor_1=2, learning_factor_2=2, seed=None):
        super().__init__(population_size, nb_features, constraints,
                         inertia=1., divergence=1.,
                         learning_factor_1=learning_factor_1,
                         learning_factor_2=learning_factor_2,
                         seed=seed)


class InertiaPSO(ParticleSwarmOptimisation):

    def __init__(self, population_size, nb_features, constraints, inertia=1.5,
                 learning_factor_1=2, learning_factor_2=2, seed=None):
        super().__init__(population_size, nb_features, constraints,
                         inertia=inertia, divergence=1.,
                         learning_factor_1=learning_factor_1,
                         learning_factor_2=learning_factor_2,
                         seed=seed)


class DivergentPSO(ParticleSwarmOptimisation):

    def __init__(self, population_size, nb_features, constraints, inertia=1.5, divergence=np.sqrt(2),
                 learning_factor_1=2, learning_factor_2=2, seed=None):
        super().__init__(population_size, nb_features, constraints,
                         inertia=inertia, divergence=divergence,
                         learning_factor_1=learning_factor_1,
                         learning_factor_2=learning_factor_2,
                         seed=seed)
