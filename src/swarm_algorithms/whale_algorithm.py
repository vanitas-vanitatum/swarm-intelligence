from src.swarm_algorithms.swarm_algorithm import SwarmIntelligence

import numpy as np
import scipy.spatial.distance as dist


class WhaleAlgorithm(SwarmIntelligence):
    def __init__(self, population_size, nb_features, constraints, attenuation_of_medium,
                 intensity_at_source, seed=0xCAFFE):
        super().__init__(population_size, nb_features, constraints, seed)
        self._eta = attenuation_of_medium
        self._rho_zero = intensity_at_source

    def get_new_positions(self, step_number):
        nearest_best_whales, distances_to_best_whales = self._get_nearest_best_whale_and_distance(self.population)
        return self.population + self._rng.uniform(0, self._rho_zero * np.exp(
            -self._eta * distances_to_best_whales[:, np.newaxis]), size=(self.population_size, self.nb_features)) * (
                       self.population[nearest_best_whales] - self.population)

    def update_positions(self, new_positions, step_number):
        self.population = new_positions
        return self.population

    def go_swarm_go(self):
        # TODO: Maybe implement in the future
        pass

    def _get_nearest_best_whale_and_distance(self, population):
        """
        Calculates nearest best whale for every whale in the population.
        :param population: Population to consider.
        :return: Indices of best whales in the population for each whale as np.ndarray N shape
        """
        assert self.is_compiled
        fit_values = self.fit_function(population)
        nearest_best_whales = []
        distances_to_best_whales = []
        for i in range(len(fit_values)):
            where_better = np.where(fit_values >= fit_values[i][0])[0]
            considered_whales = population[where_better]
            distances = dist.cdist(considered_whales, population[i:i + 1])[1:, 0]
            if len(distances) == 0:
                nearest_best_whales.append(i)
                distances_to_best_whales.append(0)
                continue
            index = np.argmax(distances)
            distances_to_best_whales.append(distances[index])
            nearest_best_whales.append(index)
        nearest_best_whales = np.array(nearest_best_whales)
        distances_to_best_whales = np.array(distances_to_best_whales)
        return nearest_best_whales, distances_to_best_whales

    @staticmethod
    def get_optimal_eta_and_rho_zero(search_space_boundaries):
        """
        Generates \eta parameter, optimal according to https://arxiv.org/pdf/1702.03389.pdf.
        :param search_space_boundaries:
            Search space for each dimension. It should be in form of list of tuples: [(b_{min}, b_{max}), ...],
            where num of tuples is equal to number of dimensions.
        :return: Optimal \eta parameter.
        """
        search_space_boundaries = np.array(search_space_boundaries)
        min_vector = search_space_boundaries[:, 0]
        max_vector = search_space_boundaries[:, 1]
        d_max = np.linalg.norm(max_vector - min_vector, ord=2)
        return -20 * np.log(0.25) / d_max, 2
