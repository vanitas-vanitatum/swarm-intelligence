from src.callbacks import PrintLogCallback, RoomDrawerCallback
from src.constraints import RoomConstraint
from src.experiments.utils import SAMPLES_PATH
from src.furnishing.room import RoomDrawer
from src.furnishing.room_utils import get_example_room, load_example_positions_for_example_room
from src.stop_conditions import EarlyStoppingCondition
from src.swarm_algorithms.quantum_algorithm import QuantumDeltaParticleSwarmOptimization
from src.swarm_algorithms.particle_swarm_optimisation import DivergentPSO
from src.swarm_algorithms.whale_algorithm import WhaleAlgorithm
from src.test_functions.furnishing import RoomFitness

room = get_example_room()
templates = load_example_positions_for_example_room(SAMPLES_PATH)
# templates = room.normalize_templates(templates)
print(room.are_furniture_ok())
rfit = RoomFitness(room)

drawer = RoomDrawer(room)
drawer.draw_all(
    tv_tv='yellow'
)

bound = [(0, 1), (0, 1), (-1, 1)] * len(room.params_to_optimize)
func = rfit
eta, rho = WhaleAlgorithm.get_optimal_eta_and_rho_zero(bound)
# alg = WhaleAlgorithm(20, len(room.params_to_optimize.flatten()), RoomConstraint(room), eta,  rho)
alg = QuantumDeltaParticleSwarmOptimization(100, len(room.params_to_optimize.flatten()), RoomConstraint(room), 1)
# alg = DivergentPSO(20, len(room.params_to_optimize.flatten()), RoomConstraint(room), 2.5, 0.7289, 2, 2)
spawn_bound = [(0, 1), (0, 1), (-0.5, 0.5)] * len(room.params_to_optimize)
alg.compile(func.fitness_function, spawn_bound, templates)
# alg.compile(func.fitness_function, spawn_bound)
print(alg.current_global_fitness)
sol = alg.go_swarm_go((EarlyStoppingCondition(30)),
                      [PrintLogCallback(), RoomDrawerCallback(room, 'temp')])
print(sol)
print(func.fitness_function(sol.reshape(1, -1)))

old_sol = room.params_to_optimize
room.apply_feature_vector(sol)
print(room.get_possible_carpet_radius())
room.update_carpet_diameter()
drawer = RoomDrawer(room)
drawer.draw_all(
    tv_tv='yellow'
)
room.apply_feature_vector(old_sol)
