import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from src.callbacks import ParamStatsStoreCallback, PrintLogCallback, PandasLogCallback
from src.constraints import LessThanConstraint, MoreThanConstraint
from src.experiments.utils import get_experiment_dir
from src.stop_conditions import EarlyStoppingCondition, StepsNumberStopCondition, AtLeastCondition, CustomStopCondition
from src.swarm_algorithms import *
from src.test_functions import (Rosenbrock, Michalewicz, Zakharov,
                                StyblinskiTang, Shwefel, Ackley)


TESTED_DIMENSIONALITY = 2

OPTIMAL_QSO = (lambda population_size, nb_features, constraints:
               QuantumDeltaParticleSwarmOptimization(population_size, nb_features, constraints,
                                                     delta_potential_length_parameter=5.3105))

OPTIMAL_PSO = (lambda population_size, nb_features, constraints:
               ParticleSwarmOptimisation(population_size, nb_features, constraints,
                                         learning_factor_1=3.0, learning_factor_2=4.0,
                                         inertia=0.6158, divergence=0.7289))

OPTIMAL_WHALE = (lambda population_size, nb_features, constraints:
                 WhaleAlgorithm(population_size, nb_features, constraints,
                                attenuation_of_medium=8.9579,
                                intensity_at_source=1.6632))

POPULATION_SIZES = np.linspace(10, 200, 20, dtype=np.int)


POPULATION_SIZE_TO_TEST_RUN_LOG = 30


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
    Shwefel(),
    Ackley()
]

FUNCTIONS_BOUNDARIES = [
    get_boundaries_and_constraints(-5, 10)[0],
    get_boundaries_and_constraints(0, np.pi)[0],
    get_boundaries_and_constraints(-5, 10)[0],
    get_boundaries_and_constraints(-5, 5)[0],
    get_boundaries_and_constraints(-500, 500)[0],
    get_boundaries_and_constraints(-32.768, 32.768)[0]
]

FUNCTIONS_CONSTRAINTS = [
    get_boundaries_and_constraints(-5, 10)[1],
    get_boundaries_and_constraints(0, np.pi)[1],
    get_boundaries_and_constraints(-5, 10)[1],
    get_boundaries_and_constraints(-5, 5)[1],
    get_boundaries_and_constraints(-500, 500)[1],
    get_boundaries_and_constraints(-32.768, 32.768)[1]
]

MAX_STEPS_NUMBER = 300
BASE_STOP_CONDITION = lambda: (StepsNumberStopCondition(MAX_STEPS_NUMBER)
                               .maybe(EarlyStoppingCondition(30)))
                               #.und(CustomStopCondition(lambda alg: not np.isinf(alg.current_global_fitness))))
TOLERANCE = 1e-6
RUN_TIMES = 10


def test_population_size(alg_constructor, nb_dimensions, functions, boundaries, constraints):

        logger = pd.DataFrame(columns=['func_name', 'population_size',
                                       'steps_number_mean', 'steps_number_std',
                                       'fitness_mean', 'fitness_std'])
        best_run_logs = OrderedDict()
        for func, bound, constr in zip(functions, boundaries, constraints):
            for pop_size in POPULATION_SIZES:
                nbs_steps = []
                fitnesses = []
                run_logs = []
                for _ in range(RUN_TIMES):
                    alg = alg_constructor(pop_size, nb_dimensions, constr)
                    alg.compile(func.fitness_function, bound)
                    stats_callback = ParamStatsStoreCallback()
                    pandas_callback = PandasLogCallback()
                    if func.minimal_value(nb_dimensions) is not None:
                        stop_condition = (BASE_STOP_CONDITION()
                                          .maybe(AtLeastCondition(func.minimal_value(nb_dimensions)
                                                                  + TOLERANCE)))
                    else:
                        stop_condition = BASE_STOP_CONDITION()
                    res = alg.go_swarm_go(stop_condition, [stats_callback, pandas_callback])
                    epoch, best, worst, avg = stats_callback.get_params()

                    nbs_steps.append(epoch)
                    fitnesses.append(best)
                    if pop_size == POPULATION_SIZE_TO_TEST_RUN_LOG:
                        run_logs.append(pandas_callback.get_log())

                print(alg.__class__.__name__, func.__class__.__name__, pop_size, np.mean(nbs_steps))
                logger = logger.append({'func_name': func.__class__.__name__,
                                        'population_size': pop_size,
                                        'steps_number_mean': np.mean(nbs_steps),
                                        'steps_number_std': np.std(nbs_steps),
                                        'fitness_mean': np.mean(fitnesses),
                                        'fitness_std': np.std(fitnesses)},
                                       ignore_index=True)
                if POPULATION_SIZE_TO_TEST_RUN_LOG == pop_size:
                    best_run_logs[func.__class__.__name__] = run_logs[np.argmin(fitnesses)]
        logger = logger.round(4)
        yield logger, best_run_logs


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
        for logger, best_run_logs in test_method(alg_constructor, nb_dimensions, functions, boundaries, constraints):
            logger.to_csv(os.path.join(csv_dir, f'{name}-{test_name}.csv'), index=False,
                          float_format='%.4f')
            logger.to_latex(os.path.join(latex_dir, f'{name}-{test_name}.tex'), index=False,
                            float_format='%.4f')

            for func, log in best_run_logs.items():
                log.to_csv(os.path.join(csv_dir, f'optimisation_log-{func}-{name}.csv'), index=False,
                           float_format='%.4f')
                log.to_latex(os.path.join(latex_dir, f'optimisation_log-{func}-{name}.tex'), index=False,
                             float_format='%.4f')
    print()


if __name__ == '__main__':
    test()
    #test(test_population_size)
