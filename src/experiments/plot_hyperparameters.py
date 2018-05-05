import argparse
import glob
import os.path as osp

import pandas as pd
import matplotlib.pyplot as plt

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('csvdir', help='Path to all csvs')
args = argument_parser.parse_args()


def plot():
    dir = args.csvdir
    y_axis = 'Fitness function value'

    for f in glob.glob(osp.join(dir, '*.csv')):
        file_name = osp.basename(f)[:-4] + '.png'
        dataframe = pd.read_csv(f)
        output_file = osp.join(osp.dirname(dir), 'plots', file_name)
        param_name = dataframe['param name'].unique()[0]
        param_name = ' '.join(x.capitalize() for x in param_name.split('_'))
        param_values = dataframe['param value']
        best = dataframe['best']
        worst = dataframe['worst']
        avg = dataframe['avg']

        plt.figure()
        plt.ylabel(y_axis)
        plt.xlabel(param_name)
        plt.plot(param_values, best, 'o--', label='Best')
        plt.plot(param_values, worst, 'o--', label='Worst')
        plt.plot(param_values, avg, 'o--', label='Average')
        plt.legend(loc='best')

        plt.savefig(output_file)


if __name__ == '__main__':
    plot()
