from src.callbacks import PrintLogCallback
from src.constraints import LessThanConstraint, MoreThanConstraint
from src.stop_conditions import EarlyStoppingCondition
from src.swarm_algorithms import *
from src.test_functions import *
from src.callbacks import Drawer2d

constraints = LessThanConstraint(37).und(MoreThanConstraint(-37))
nb_features = 2

bound = [(-37, 37), (-37, 37)]
func = Griewank()
eta, rho = WhaleAlgorithm.get_optimal_eta_and_rho_zero(bound)
# alg = WhaleAlgorithm(20, len(room.params_to_optimize.flatten()), RoomConstraint(room), eta,  rho)
alg = QuantumDeltaParticleSwarmOptimization(100, nb_features, constraints, 1)
# alg = DivergentPSO(20, nb_features, constraints, 2.5, 0.7289, 2, 2)
alg.compile(func.fitness_function, bound)
print(alg.current_global_fitness)
sol = alg.go_swarm_go((EarlyStoppingCondition(100)),
                      [PrintLogCallback(), Drawer2d(bound, 1000, 0.4)])
