import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import os.path as osp
from matplotlib import cm
from src.furnishing.room import RoomDrawer

# from collections import OrderedDict

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


class Callback:

    def __init__(self):
        self.swarm_algorithm = None

    def initialize_callback(self, swarm_algorithm):
        self.swarm_algorithm = swarm_algorithm

    def on_optimization_start(self):
        pass

    def on_optimization_end(self):
        pass

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        pass


class CallbackContainer(Callback):
    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = callbacks if callbacks else []

    def __iter__(self):
        for x in self.callbacks:
            yield x

    def __len__(self):
        return len(self.callbacks)

    def initialize_callback(self, swarm_algorithm):
        for callback in self.callbacks:
            callback.initialize_callback(swarm_algorithm)

    def on_optimization_start(self):
        for callback in self.callbacks:
            callback.on_optimization_start()

    def on_optimization_end(self):
        for callback in self.callbacks:
            callback.on_optimization_end()

    def on_epoch_start(self):
        for callback in self.callbacks:
            callback.on_epoch_start()

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end()


class Drawer2d(Callback):
    def __init__(self, space_boundaries, space_sampling_size=1000,
                 isolines_spacing=4):
        super().__init__()
        self.optimized_function = None
        self.space_sampling_size = space_sampling_size
        (self.x1, self.x2), (self.y1, self.y2) = space_boundaries
        self.last_population = None

        self.fig = None
        self.ax = None
        self.space_visualization_coordinates = None
        self.contour_values = None
        self.isolines_spacing = isolines_spacing

    def initialize_callback(self, swarm_algorithm):
        super().initialize_callback(swarm_algorithm)
        self.optimized_function = swarm_algorithm.fit_function

        x = np.linspace(self.x1, self.x2, self.space_sampling_size)
        y = np.linspace(self.y1, self.y2, self.space_sampling_size)

        self.space_visualization_coordinates = np.stack(np.meshgrid(x, y))
        self.contour_values = self.optimized_function(
            self.space_visualization_coordinates.reshape(2, -1).T
        ).reshape(self.space_sampling_size, self.space_sampling_size)

    def on_optimization_start(self):
        plt.ion()

    def on_epoch_end(self):
        super().on_epoch_end()

        population = self.swarm_algorithm.population

        xs = population[:, 0]
        ys = population[:, 1]

        dots_coordinates = np.stack(np.meshgrid(xs, ys))
        plt.contour(
            self.space_visualization_coordinates[0],
            self.space_visualization_coordinates[1],
            self.contour_values,
            cmap=cm.coolwarm,
            levels=np.arange(
                np.min(self.contour_values).astype(np.float16),
                np.max(self.contour_values).astype(np.float16),
                self.isolines_spacing
            ),
            zorder=1
        )

        plt.scatter(

            xs,
            ys,
            marker='x',
            linewidths=2,
            color='red',
            s=100,
            zorder=2
        )

        if self.last_population is not None:
            for i in range(len(population)):
                pos = self.last_population[i]
                new_pos = population[i]
                dx, dy = new_pos - pos
                x, y = pos

                plt.arrow(x, y, dx, dy, head_width=0.5,
                          head_length=1, fc='k', ec='k')

        plt.pause(0.01)
        self.last_population = population
        plt.clf()
        plt.cla()

    def on_optimization_end(self):
        plt.ioff()


class PrintLogCallback(Callback):

    def on_epoch_end(self):
        print('Epoch:', self.swarm_algorithm._step_number,
              'Global Best:', self.swarm_algorithm.current_global_fitness)


class FileLogCallback(Callback):
    NON_HYPERPARAMS = ['population', 'population_size',
                       '_compiled', '_seed',
                       '_rng', '_step_number',
                       'fit_function',
                       'global_best_solution',
                       'local_best_solutions',
                       'nb_features',
                       'constraints',
                       'current_global_fitness',
                       'current_local_fitness']

    def __init__(self, result_filename):
        super().__init__()
        self.log_df = pd.DataFrame(columns=['Epoch', 'Best Global Fitness', 'Worst Local Fitness'])
        self.result_filename = result_filename

    def on_epoch_end(self):
        epoch = int(self.swarm_algorithm._step_number)
        bgfit = self.swarm_algorithm.current_global_fitness
        wlfit = np.max(self.swarm_algorithm.current_local_fitness)

        self.log_df = self.log_df.append({'Epoch': epoch,
                                          'Best Global Fitness': bgfit,
                                          'Worst Local Fitness': wlfit},
                                         ignore_index=True)

    def on_optimization_end(self):
        meta = {'FitFunction': self.swarm_algorithm.fit_function.__self__.__class__.__name__,
                'Algorithm': self.swarm_algorithm.__class__.__name__,
                'PopulationSize': self.swarm_algorithm.population_size,
                'NbFeatures': self.swarm_algorithm.nb_features}

        hyperparams = self.swarm_algorithm.__dict__.copy()
        for k in self.NON_HYPERPARAMS:
            hyperparams.pop(k)

        for k in hyperparams:
            hyperparams[k] = str(hyperparams[k])

        meta['AlgorithmHyperparams'] = hyperparams
        with open(self.result_filename + '-meta.yaml', 'w') as f:
            yaml.dump(meta, f, default_flow_style=False)

        self.log_df['Epoch'] = pd.to_numeric(self.log_df['Epoch'], downcast='integer')
        self.log_df.to_csv(self.result_filename + '-log.csv', index=False)


class ParamStatsStoreCallback(Callback):
    def __init__(self):
        super().__init__()
        self.last_epoch = 0
        self.best_fitness = np.inf
        self.worst_fitness = -np.inf
        self.average_fitness = np.inf

    def on_epoch_end(self):
        self.last_epoch = int(self.swarm_algorithm._step_number)
        self.best_fitness = min(self.swarm_algorithm.current_global_fitness, self.best_fitness)
        self.worst_fitness = max(np.max(self.swarm_algorithm.current_local_fitness), self.worst_fitness)
        self.average_fitness = min(self.average_fitness, np.mean(self.swarm_algorithm.current_local_fitness))

    def get_params(self):
        return [self.last_epoch, self.best_fitness, self.worst_fitness, self.average_fitness]


class EpochLoggerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.logger = pd.DataFrame(columns=['epoch', 'best', 'avg', 'worst'])

    def on_epoch_end(self):
        self.logger.append({
            'epoch': self.swarm_algorithm._step_number,
            'best': self.swarm_algorithm.current_global_fitness,
            'avg': np.mean(self.swarm_algorithm.current_local_fitness),
            'worst': np.max(self.swarm_algorithm.current_local_fitness)
        })


class RoomDrawerCallback(Callback):
    def __init__(self, room, folder_output, epoch_break=4):
        super().__init__()
        self.room = room
        self.epoch_break = epoch_break
        self.counter = 0
        self.folder_output = folder_output
        plt.ion()
        self.drawer = RoomDrawer(room)

    def on_optimization_start(self):
        self.drawer.draw_all(tv_tv='yellow')

    def on_epoch_end(self):
        if self.counter % self.epoch_break == 0:

            old_params = self.room.params_to_optimize
            new_params = self.swarm_algorithm.global_best_solution
            self.room.apply_feature_vector(new_params)
            self.room.update_carpet_diameter()
            self.drawer.update(tv_tv='yellow')
            self.drawer.figure.savefig(osp.join(self.folder_output, f'{self.counter}.png'))
            self.room.apply_feature_vector(old_params)
            self.room.update_carpet_diameter()
            self.drawer.clear()
        self.counter += 1

