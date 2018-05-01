import numpy as np

class RoomFitness:
    def __init__(self, room, simple=False):
        self.room = room
        self.simple = simple

    def fitness_function(self, solutions):
        old_params = self.room.params_to_optimize
        fitnesses = np.empty((solutions.shape[0], 1))
        for i in range(solutions.shape[0]):
            sol = solutions[i, :]
            if self.simple:
                for j in range(2,sol.shape[0],3):
                    sol[j] = 0
            self.room.apply_feature_vector(sol)
            fitnesses[i, 0] = -self.room.get_possible_carpet_radius()

        self.room.apply_feature_vector(old_params)
        return fitnesses
