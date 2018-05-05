import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from src.callbacks import ParamStatsStoreCallback
from src.constraints import LessThanConstraint, MoreThanConstraint
from src.experiments.utils import get_experiment_dir
from src.stop_conditions import StepsNumberStopCondition
from src.swarm_algorithms import *
from src.test_functions import Ackley

QSO_PRAMS = {
    'population_size': np.linspace(10, 200, 20, dtype=np.int),
    'delta_potential_length_parameter': np.linspace(0.1, 10, 20)
}

PSO_PARAMS = {
    'population_size': np.linspace(10, 200, 20, dtype=np.int),
    'inertia': np.linspace(0.1, 5, 20),
    'divergence': np.linspace(0.1, 5, 20),
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
POPULATION_SIZE = 50


def test_single(algorithm_constructor, boundaries, params, default_params):
    for key, all_values in params.items():
        logger = pd.DataFrame(columns=['epoch', 'best', 'avg', 'worst', 'param value'])
        for value in all_values:
            alg = algorithm_constructor(**default_params)
            setattr(alg, key, value)
            alg.compile(TESTED_FUNCTION.fitness_function, boundaries)

            stats_callback = ParamStatsStoreCallback()
            alg.go_swarm_go(StepsNumberStopCondition(STEPS_NUMBER), [stats_callback])
            epoch, best, worst, avg = stats_callback.get_params()
            logger = logger.append({'epoch': epoch, 'best': best, 'avg': avg, 'worst': worst, 'param name': key,
                                    'param value': value},
                                   ignore_index=True)
        logger = logger.round(4)
        yield logger, key


def test():
    d = [2, 10]
    names = ['PSO', 'Whale', 'QSO']
    algs_params = OrderedDict({
        ParticleSwarmOptimisation: PSO_PARAMS,
        WhaleAlgorithm: WHALE_PARAMS,
        QuantumDeltaParticleSwarmOptimization: QSO_PRAMS
    })
    experiment_dir, csv_dir, latex_dir, plots_dir = get_experiment_dir()
    constraints = MoreThanConstraint(-32.768).und(LessThanConstraint(32.768))
    for name, (alg, params) in zip(names, algs_params.items()):
        print(f'\r{alg}', end='')
        for dimension in d:
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
            for logger, param_name in test_single(alg, boundaries, params, default_params):
                logger.to_csv(os.path.join(csv_dir, f'{name}-{param_name}-{dimension}.csv'), index=False,
                              float_format='%.4f')
                logger.to_latex(os.path.join(latex_dir, f'{name}-{param_name}-{dimension}.csv'), index=False,
                                float_format='%.4f')
    print()


if __name__ == '__main__':
    test()
