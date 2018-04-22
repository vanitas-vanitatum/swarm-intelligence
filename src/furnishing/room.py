import matplotlib.pyplot as plt
import numpy as np
from descartes import PolygonPatch
from shapely.geometry import Polygon

from src.furnishing.drawable import Drawable
from src.furnishing.furniture_collection import Tv


class Room(Drawable):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.furniture = []
        points = np.array([
            [0, 0],
            [self.width, 0],
            [self.width, self.height],
            [0, self.height]
        ])
        self.shape = Polygon(points)

    def get_patch(self, **kwargs):
        return PolygonPatch(self.shape, fc=kwargs.get('color', '#dddddd'),
                            ec=kwargs.get('color', '#111111'),
                            zorder=-2)

    def add_furniture(self, *furniture):
        for f in furniture:
            f.room = self
            self.furniture.append(f)

    def are_all_furniture_inside(self):
        for f in self.furniture:
            self.is_furniture_inside(f)

    def is_furniture_inside(self, furniture):
        return self.shape.contains(furniture)

    def has_tv(self):
        return any([isinstance(f, Tv) for f in self.furniture])

    def get_tv(self):
        for f in self.furniture:
            if isinstance(f, Tv):
                return f
        return None

    @property
    def params_to_optimize(self):
        params = []
        for f in self.furniture:
            if f.is_optimasible:
                params.append(f.params_to_optimize)
        return np.array(params)


class RoomDrawer:
    def __init__(self, room, figsize=(16, 16)):
        self.room = room
        self.figure = plt.figure(figsize=figsize)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(xmin=0, xmax=self.room.width)
        self.ax.set_ylim(ymin=0, ymax=self.room.height)

    def draw_all(self, **kwargs):
        self.room.draw(self.ax)
        for furniture in self.room.furniture:
            furniture.draw(self.ax, **kwargs)
        plt.show()
