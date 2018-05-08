import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from src.callbacks import ParamStatsStoreCallback
from src.constraints import LessThanConstraint, MoreThanConstraint
from src.experiments.utils import get_experiment_dir
from src.stop_conditions import EarlyStoppingCondition
from src.swarm_algorithms import *
from src.test_functions import Ackley

TESTED_DIMENSIONALITIES = [2, 10]

QSO_PRAMS = {
    'population_size': np.linspace(10, 200, 20, dtype=np.int),
    'delta_potential_length_parameter': np.linspace(0.1, 10, 20)
}

PSO_PARAMS = {
    'population_size': np.linspace(10, 200, 20, dtype=np.int),
    'inertia': np.linspace(0.1, 5, 20),
    'divergence': np.asarray(sorted(list(np.linspace(0.1, 5, 20))+[0.7289])),
    'learning_factor_1': np.linspace(0.5, 10, 20),
    'learning_factor_2': np.linspace(0.5, 10, 20),

}

WHALE_PARAMS = {
    'population_size': np.linspace(10, 200, 20, dtype=np.int),
    'attenuation': np.linspace(0.1, 10, 20),
    'intensity_at_source': np.linspace(0.1, 10, 20)
}

TESTED_FUNCTION = Ackley()
STEPS_NUMBER = 300
POPULATION_SIZE = 30
RUN_TIMES = 10


def test_single(algorithm_constructor, constraints, params):
    for key, all_values in params.items():
        logger = pd.DataFrame(columns=['param_name', 'param_value',
                                       'dimension2_mean', 'dimension2_std',
                                       'dimension10_mean', 'dimension10_std'])
        for value in all_values:
            best_dim_2, best_dim_10 = [], []
            for _ in range(RUN_TIMES):
                best, epoch, worst, avg, alg = run_with_dimensionality(algorithm_constructor, constraints, 2, key, value)
                best_dim_2.append(epoch * best)

                best, epoch, worst, avg, alg = run_with_dimensionality(algorithm_constructor, constraints, 10, key, value)
                best_dim_10.append(epoch * best)

            logger = logger.append({'param_name': key,
                                    'param_value': value,
                                    'dimension2_mean': np.mean(best_dim_2),
                                    'dimension2_std': np.std(best_dim_2),
                                    'dimension10_mean': np.mean(best_dim_10),
                                    'dimension10_std': np.std(best_dim_10)},
                                   ignore_index=True)
        logger = logger.round(4)
        yield logger, key


def test_steps_number(algorithm_constructor, constraints, params):
    for key, all_values in params.items():
        logger = pd.DataFrame(columns=['param_name', 'param_value',
                                       'dimension2_mean', 'dimension2_std',
                                       'dimension10_mean', 'dimension10_std'])
        for value in all_values:
            best_dim_2, best_dim_10 = [], []
            for _ in range(RUN_TIMES):
                best, epoch, worst, avg, alg = run_with_dimensionality(algorithm_constructor, constraints, 2, key, value)
                best_dim_2.append(alg._step_number)

                best, epoch, worst, avg, alg = run_with_dimensionality(algorithm_constructor, constraints, 10, key, value)
                best_dim_10.append(alg._step_number)

            logger = logger.append({'param_name': key,
                                    'param_value': value,
                                    'dimension2_mean': np.mean(best_dim_2),
                                    'dimension2_std': np.std(best_dim_2),
                                    'dimension10_mean': np.mean(best_dim_10),
                                    'dimension10_std': np.std(best_dim_10)},
                                   ignore_index=True)
        logger = logger.round(4)
        yield logger, key


def run_with_dimensionality(algorithm_constructor, constraints, dim, key, value):
    default_params, boundaries = get_params_and_boundaries(dim, constraints, algorithm_constructor)
    alg = algorithm_constructor(**default_params)
    setattr(alg, key, value)
    alg.compile(TESTED_FUNCTION.fitness_function, boundaries)
    stats_callback = ParamStatsStoreCallback()
    alg.go_swarm_go(EarlyStoppingCondition(30), [stats_callback])
    epoch, best, worst, avg = stats_callback.get_params()
    return best, epoch, worst, avg, alg


def get_params_and_boundaries(dimension, constraints, alg):
    boundaries = np.zeros((dimension, 2))
    boundaries[:, 0] = -32.768
    boundaries[:, 1] = 32.768
    if alg == ParticleSwarmOptimisation:
        default_params = dict(
            population_size=POPULATION_SIZE,
            nb_features=dimension,
            constraints=constraints,
            inertia=1.,
            divergence=1.,
            learning_factor_1=2.,
            learning_factor_2=2.,
            seed=None
        )
    elif alg == WhaleAlgorithm:
        rho, eta = WhaleAlgorithm.get_optimal_eta_and_rho_zero(boundaries)
        default_params = dict(
            population_size=POPULATION_SIZE,
            nb_features=dimension,
            constraints=constraints,
            attenuation_of_medium=eta,
            intensity_at_source=rho,
            seed=None
        )
    else:
        default_params = dict(
            population_size=POPULATION_SIZE,
            nb_features=dimension,
            constraints=constraints,
            delta_potential_length_parameter=1,
            seed=None
        )
    return default_params, boundaries


def test(test_method=test_single):
    names = ['PSO',
             'Whale',
             'QSO'
             ]
    algs_params = OrderedDict({
        ParticleSwarmOptimisation: PSO_PARAMS,
        WhaleAlgorithm: WHALE_PARAMS,
        QuantumDeltaParticleSwarmOptimization: QSO_PRAMS
    })
    experiment_dir, csv_dir, latex_dir, plots_dir = get_experiment_dir()
    constraints = MoreThanConstraint(-32.768).und(LessThanConstraint(32.768))

    if WhaleAlgorithm in algs_params.keys():
        rho2, eta2 = (WhaleAlgorithm
                    .get_optimal_eta_and_rho_zero(get_params_and_boundaries(2, constraints, WhaleAlgorithm)[1]))
        rho10, eta10 = (WhaleAlgorithm
            .get_optimal_eta_and_rho_zero(get_params_and_boundaries(10, constraints, WhaleAlgorithm)[1]))
        algs_params[WhaleAlgorithm]['attenuation'] = np.concatenate(
            [algs_params[WhaleAlgorithm]['attenuation'], [eta2, eta10]])
        algs_params[WhaleAlgorithm]['intensity_at_source'] = np.concatenate(
            [algs_params[WhaleAlgorithm]['intensity_at_source'], [rho2, rho10]])

    for name, (alg, params) in zip(names, algs_params.items()):
        print(f'\r{alg}', end='')
        for logger, param_name in test_method(alg, constraints, params):
            postfix = '' if test_method == test_single else '-'+str(test_method.__name__)
            logger.to_csv(os.path.join(csv_dir, f'{name}-{param_name}{postfix}.csv'), index=False,
                          float_format='%.4f')
            logger.to_latex(os.path.join(latex_dir, f'{name}-{param_name}{postfix}.tex'), index=False,
                            float_format='%.4f')
    print()


if __name__ == '__main__':
    test()
    #test(test_steps_number)
