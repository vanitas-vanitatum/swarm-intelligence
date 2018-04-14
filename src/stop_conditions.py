import numpy as np
import math
from src.specification import Specification


class CustomStopCondition(Specification):

    def __init__(self, check_function):
        self.check_function = check_function

    def check(self, swarm_algorithm):
        return self.check_function(swarm_algorithm)


class StepsNumberStopCondition(Specification):

    def __init__(self, nb_steps):
        self.nb_steps = nb_steps

    def check(self, swarm_algorithm):
        return swarm_algorithm._step_number >= self.nb_steps


class EarlyStoppingCondition(Specification):

    def __init__(self, patience):
        self.patience = patience
        self.nb_steps_without_improvement = 0
        self.so_far_best_fitness = np.inf

    def check(self, swarm_algorithm):
        if swarm_algorithm.current_global_fitness < self.so_far_best_fitness:
            self.so_far_best_fitness = swarm_algorithm.current_global_fitness
            self.nb_steps_without_improvement = 0
        else:
            self.nb_steps_without_improvement += 1

        return self.nb_steps_without_improvement > self.patience

