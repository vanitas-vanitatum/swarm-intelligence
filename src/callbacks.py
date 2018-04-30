from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib

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
