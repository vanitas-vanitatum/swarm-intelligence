from src.constraints import NoneConstraint, CustomConstraint, LessThanConstraint, MoreThanConstraint, RoomConstraint
from src.swarm_algorithms.particle_swarm_optimisation import DivergentPSO, ParticleSwarmOptimisation
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

#FUN = 'Ackley'
FUN = 'Levy'
#FUN = 'Micha'
#FUN = 'Schwefel'
#FUN = 'Griewank'

ALG = 'PSO'

arrows = True
if FUN == 'Levy':
    func = Levy()
    bound = [(-10,10)]*2
    constr = LessThanConstraint(10).und(MoreThanConstraint(-10))
    sampling = 100
    isolines = 1
elif FUN == 'Micha':
    func = Michalewicz()
    bound = [(0,4)]*2
    constr = LessThanConstraint(4).und(MoreThanConstraint(0))
    sampling = 100
    isolines = 0.1
    arrows=False
elif FUN == 'Schwefel':
    func = Shwefel()
    bound = [(-500,500)]*2
    constr = LessThanConstraint(500).und(MoreThanConstraint(-500))
    sampling = 300
    isolines = 100
elif FUN == 'Griewank':
    func = Griewank()
    bound = [(-50,50)]*2
    constr = LessThanConstraint(50).und(MoreThanConstraint(-50))
    sampling = 300
    isolines = 0.1
else:
    func = Ackley()
    bound = [(-37,37),(-37,37)]
    constr = LessThanConstraint(37).und(MoreThanConstraint(-37))
    sampling = 100
    isolines = 1


eta, rho = WhaleAlgorithm.get_optimal_eta_and_rho_zero(bound)

if ALG == 'PSO':
    alg = ParticleSwarmOptimisation(10, 2, constr, 0.5, 0.7289, 2, 2)
elif ALG == 'Whale':
    alg = WhaleAlgorithm(10, 2, constr, eta, rho)
else:
    alg = QuantumDeltaParticleSwarmOptimization(10, 2, constr, 5)

alg.compile(func.fitness_function, bound)
print(alg.current_global_fitness)
sol = alg.go_swarm_go(StepsNumberStopCondition(50)
                      .maybe(EarlyStoppingCondition(10)),
                      [PrintLogCallback(), Drawer2d(bound, sampling, isolines, arrows)])
print(sol)
print(func.fitness_function(sol.reshape(1,-1)))
