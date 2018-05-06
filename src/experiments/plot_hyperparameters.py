import glob
import os.path as osp

import click
import matplotlib.pyplot as plt
import pandas as pd


@click.group()
def cli():
    pass


@cli.command()
@click.argument('csv_dir', type=click.Path(exists=True))
def plot_hyperparams(csv_dir):
    dir = csv_dir
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


@cli.command()
@click.command('csv_path', type=click.Path(exists=True))
@click.command('output_path')
def plot_comparison(csv_path, output_path):
    algorithm_names = ['PSO', 'Whale', 'QSO']
    plt.figure()
    plt.ylabel('Negated carpet radius')
    plt.xlabel('Epoch')

    data = pd.read_csv(csv_path)
    for name in algorithm_names:
        algorithm_data = data.loc[data['algorithm'] == name]
        plt.plot(algorithm_data['epochs'], algorithm_data['carpet_size'])
    plt.legend(loc='best')
    plt.savefig(output_path)


if __name__ == '__main__':
    cli()
