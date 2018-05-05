import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from src.callbacks import ParamStatsStoreCallback, PrintLogCallback
from src.constraints import LessThanConstraint, MoreThanConstraint
from src.experiments.utils import get_experiment_dir
from src.stop_conditions import EarlyStoppingCondition, StepsNumberStopCondition, AtLeastCondition, CustomStopCondition
from src.swarm_algorithms import *
from src.test_functions import (Rosenbrock, Michalewicz, Zakharov,
                                StyblinskiTang, Shwefel)


TESTED_DIMENSIONALITY = 2

OPTIMAL_QSO = (lambda population_size, nb_features, constraints:
               QuantumDeltaParticleSwarmOptimization(population_size, nb_features, constraints,
                                                     delta_potential_length_parameter=7.9158))

OPTIMAL_PSO = (lambda population_size, nb_features, constraints:
               ParticleSwarmOptimisation(population_size, nb_features, constraints,
                                         learning_factor_1=1.5, learning_factor_2=9.5,
                                         inertia=0.3579, divergence=0.7289))

OPTIMAL_WHALE = (lambda population_size, nb_features, constraints:
                 WhaleAlgorithm(population_size, nb_features, constraints,
                                attenuation_of_medium=9.4789,
                                intensity_at_source=9.4789))

POPULATION_SIZES = np.linspace(10, 200, 20, dtype=np.int)


def get_boundaries_and_constraints(min_bound, max_bound):
    boundaries = np.zeros((TESTED_DIMENSIONALITY, 2))
    boundaries[:, 0] = min_bound
    boundaries[:, 1] = max_bound
    return (boundaries,
            LessThanConstraint(max_bound)
            .und(MoreThanConstraint(min_bound)))


TESTED_FUNCTIONS = [
    Rosenbrock(),
    Michalewicz(),
    Zakharov(),
    StyblinskiTang(),
    Shwefel()
]

FUNCTIONS_BOUNDARIES = [
    get_boundaries_and_constraints(-5, 10)[0],
    get_boundaries_and_constraints(0, np.pi)[0],
    get_boundaries_and_constraints(-5, 10)[0],
    get_boundaries_and_constraints(-5, 5)[0],
    get_boundaries_and_constraints(-500, 500)[0]
]

FUNCTIONS_CONSTRAINTS = [
    get_boundaries_and_constraints(-5, 10)[1],
    get_boundaries_and_constraints(0, np.pi)[1],
    get_boundaries_and_constraints(-5, 10)[1],
    get_boundaries_and_constraints(-5, 5)[1],
    get_boundaries_and_constraints(-500, 500)[1]
]

TESTED_FUNCTION = Shwefel()
MAX_STEPS_NUMBER = 1000
BASE_STOP_CONDITION = lambda: (StepsNumberStopCondition(MAX_STEPS_NUMBER)
                               .maybe(EarlyStoppingCondition(10)).und(CustomStopCondition(lambda alg:
                                                                                          not np.isinf(alg.current_global_fitness)))
                               .und(CustomStopCondition(lambda alg: not np.isinf(alg.current_global_fitness))))


RUN_TIMES = 10


def test_population_size(alg_constructor, nb_dimensions, functions, boundaries, constraints):

        logger = pd.DataFrame(columns=['func_name', 'population_size',
                                       'steps_number_mean', 'steps_number_std',
                                       'fitness_mean', 'fitness_std'])
        for func, bound, constr in zip(functions, boundaries, constraints):
            for pop_size in POPULATION_SIZES:
                nbs_steps = []
                fitnesses = []
                for _ in range(RUN_TIMES):
                    alg = alg_constructor(pop_size, nb_dimensions, constr)
                    alg.compile(func.fitness_function, bound)
                    stats_callback = ParamStatsStoreCallback()
                    if func.minimal_value(nb_dimensions) is not None:
                        stop_condition = BASE_STOP_CONDITION().maybe(AtLeastCondition(func.minimal_value(nb_dimensions)))
                    else:
                        stop_condition = BASE_STOP_CONDITION()
                    res = alg.go_swarm_go(stop_condition, [stats_callback])
                    epoch, best, worst, avg = stats_callback.get_params()

                    nbs_steps.append(epoch)
                    fitnesses.append(best)
                logger = logger.append({'func_name': func.__class__.__name__,
                                        'population_size': pop_size,
                                        'steps_number_mean': np.mean(nbs_steps),
                                        'steps_number_std': np.std(nbs_steps),
                                        'fitness_mean': np.mean(fitnesses),
                                        'fitness_std': np.std(fitnesses)},
                                       ignore_index=True)
        logger = logger.round(4)
        yield logger


def test(test_method=test_population_size):
    algs = OrderedDict({
        'PSO': OPTIMAL_PSO,
        'Whale': OPTIMAL_WHALE,
        'QSO': OPTIMAL_QSO
    })
    nb_dimensions = TESTED_DIMENSIONALITY

    functions = TESTED_FUNCTIONS
    boundaries = FUNCTIONS_BOUNDARIES
    constraints = FUNCTIONS_CONSTRAINTS
    experiment_dir, csv_dir, latex_dir, plots_dir = get_experiment_dir(comparison_dir=True)

    for name, alg_constructor in algs.items():
        print(f'{name}')
        test_name = str(test_method.__name__)
        for logger in test_method(alg_constructor, nb_dimensions, functions, boundaries, constraints):
            logger.to_csv(os.path.join(csv_dir, f'{name}-{test_name}.csv'), index=False,
                          float_format='%.4f')
            logger.to_latex(os.path.join(latex_dir, f'{name}-{test_name}.tex'), index=False,
                            float_format='%.4f')
    print()


if __name__ == '__main__':
    #test()
    test(test_population_size)
