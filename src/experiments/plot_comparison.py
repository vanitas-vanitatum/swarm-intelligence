import argparse
import glob
import os.path as osp

import pandas as pd
import matplotlib.pyplot as plt

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('--csvdir', help='Path to all csvs')
argument_parser.add_argument('--writestd', action='store_true')
args = argument_parser.parse_args()

TESTED_FUNCTIONS = [
    'Rosenbrock',
    'Michalewicz',
    'Zakharov',
    'StyblinskiTang',
    'Shwefel',
    'Ackley'
]

TESTED_ALGORITHMS = ['PSO', 'Whale', 'QSO']
EARLY_STOPPING_PATIENCE = 30

def plot():
    dir = args.csvdir
    y_axis = 'Fitness function value'
    x_axis = 'Number of steps'
    print(args.writestd)

    for func in TESTED_FUNCTIONS:
        f = osp.join(dir, f'optimisation_log-{func}-{TESTED_ALGORITHMS[0]}.csv')
        file_name = f'optimisation_log-{func}.png'
        output_file = osp.join(osp.dirname(dir), 'plots', file_name)

        plt.figure()
        plt.ylabel(y_axis)
        plt.xlabel(x_axis)

        for alg in TESTED_ALGORITHMS:
            f = osp.join(dir, f'optimisation_log-{func}-{alg}.csv')
            dataframe = pd.read_csv(f)

            steps_numbers = dataframe['Epoch']
            fitnesses = dataframe['Best Global Fitness']
            worst_fit = dataframe['Worst Local Fitness']

            plt.errorbar(steps_numbers, fitnesses, fmt='o--', markersize=4, yerr=0, label=alg)
        print(plt.xlim())
        plt.xlim(-1, min(50, plt.xlim()[1]-EARLY_STOPPING_PATIENCE))
        print(plt.xlim())

        plt.legend(loc='best')
        plt.savefig(output_file)


def make_tables():
    dir = args.csvdir

    dfs = None
    for alg in TESTED_ALGORITHMS:
        f = osp.join(dir, f'{alg}-test_population_size.csv')
        df = pd.read_csv(f)
        df_res = df[['func_name', 'population_size']]
        df_res[(alg+'_steps_number')] = (df.steps_number_mean.astype(str).str.rjust(10, ' ')
                                         +
                                         ((r' $\pm$ ' + df.steps_number_std.astype(str).str.rjust(10, ' '))
                                          if args.writestd else ''))
        df_res[(alg + '_fitness')] = (df.fitness_mean.astype(str).str.rjust(10, ' ')
                                      +
                                      ((r' $\pm$ ' + df.fitness_std.astype(str).str.rjust(10, ' '))
                                       if args.writestd else ''))
        if dfs is None:
            dfs = df_res
        else:
            dfs = dfs.merge(df_res, on=['func_name', 'population_size'])

    cols = (['func_name', 'population_size']
            + [alg + '_steps_number' for alg in TESTED_ALGORITHMS]
            + [alg + '_fitness' for alg in TESTED_ALGORITHMS])

    dfs = dfs[cols]
    for func in TESTED_FUNCTIONS:
        df = dfs[dfs.func_name == func]

        file_name = f'{func}_population_size.tex'
        output_file = osp.join(osp.dirname(dir), 'plots', file_name)
        df.to_latex(output_file, index=False, float_format='%.4f', escape=False)






if __name__ == '__main__':
    plot()
    make_tables()