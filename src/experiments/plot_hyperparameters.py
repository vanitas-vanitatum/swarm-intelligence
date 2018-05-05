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
    y_axis = 'Fitness function value $\\times$ epoch number'

    for f in glob.glob(osp.join(dir, '*.csv')):
        file_name = osp.basename(f)[:-4] + '.png'
        dataframe = pd.read_csv(f)
        output_file = osp.join(osp.dirname(dir), 'plots', file_name)
        param_name = dataframe['param_name'].unique()[0]
        param_name = ' '.join(x for x in param_name.split('_')).capitalize()
        param_values = dataframe['param_value']
        dim2 = dataframe['dimension2_mean']
        dim10 = dataframe['dimension10_mean']

        dim2_std = dataframe['dimension2_std']
        dim10_std = dataframe['dimension10_std']

        plt.figure()
        plt.ylabel(y_axis)
        plt.xlabel(param_name)
        plt.errorbar(param_values, dim2, fmt='o--', yerr=dim2_std, label="2 dimensions")
        plt.errorbar(param_values, dim10, fmt='o--', yerr=dim10_std, label="10 dimensions")

        plt.legend(loc='best')

        plt.savefig(output_file)


if __name__ == '__main__':
    plot()
