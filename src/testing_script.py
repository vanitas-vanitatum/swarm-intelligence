from src.constraints import NoneConstraint, CustomConstraint, LessThanConstraint, MoreThanConstraint, RoomConstraint
from src.swarm_algorithms.particle_swarm_optimisation import DivergentPSO
from src.swarm_algorithms.whale_algorithm import WhaleAlgorithm
from src.swarm_algorithms.quantum_algorithm import QuantumDeltaParticleSwarmOptimization
from src.test_functions.plate_shaped import Zakharov
from src.test_functions.many_minima import Griewank, Ackley, Levy, Shwefel
from src.test_functions.steep_ridges import Michalewicz
from src.stop_conditions import AtLeastCondition,StepsNumberStopCondition, EarlyStoppingCondition
from src.callbacks import Drawer2d, PrintLogCallback, FileLogCallback
from src.furnishing.furniture_collection import *
from src.furnishing.furniture_construction import *
from src.furnishing.room import Room, RoomDrawer
from src.test_functions.furnishing import RoomFitness

room = Room(15, 15)
room.add_furniture(RectangularFurniture(7, 3, 90, 3, 3, False, True))
#room.add_furniture(Cupboard(4, 4, 180, 3, 2, False))

sofa = Sofa(7, 7, 90, 4, 2, False)
tv = Tv(10.5, 7, -45, 5, 2, 3, 1, False, name='tv')

room.add_furniture(RoundedCornersFurniture(7, 13, 35, 1, 1, True, True))
room.add_furniture(Window(15, 7, room.width, room.height, 4))
room.add_furniture(sofa)
room.add_furniture(Door(0, 7, room.width, room.height, 4))
room.add_furniture(tv)
room.update_carpet_diameter()
rfit = RoomFitness(room)

drawer = RoomDrawer(room)
drawer.draw_all(
    tv_tv='yellow'
)

bound = [(0,15),(0,15),(-360,360)]* len(room.params_to_optimize)
func = rfit
eta, rho = WhaleAlgorithm.get_optimal_eta_and_rho_zero(bound)
# alg = WhaleAlgorithm(10, len(room.params_to_optimize.flatten()), RoomConstraint(room), eta, rho)
alg = QuantumDeltaParticleSwarmOptimization(50, len(room.params_to_optimize.flatten()), RoomConstraint(room), 1)
#alg = DivergentPSO(10,2, cc, 2.5, 0.7289, 10, 2)
alg.compile(func.fitness_function, bound)
print(alg.current_global_fitness)
sol = alg.go_swarm_go(StepsNumberStopCondition(100)
                      .maybe(EarlyStoppingCondition(100)),
                      [PrintLogCallback()])
print(sol)
print(func.fitness_function(sol.reshape(1,-1)))


old_sol = room.params_to_optimize
room.apply_feature_vector(sol)
print(room.get_possible_carpet_radius())
room.update_carpet_diameter()
drawer = RoomDrawer(room)
drawer.draw_all(
    tv_tv='yellow'
)
room.apply_feature_vector(old_sol)
