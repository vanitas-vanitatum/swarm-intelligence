import matplotlib.pyplot as plt
import numpy as np
from descartes import PolygonPatch
from shapely.geometry import Polygon, Point

from src.common import ZORDERS
from src.furnishing.drawable import Drawable
from src.furnishing.furniture_collection import Tv, Carpet, Window, Door


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
        self.carpet = Carpet(width, height, 10)
        self.carpet_center = Point(np.array([width / 2, height / 2]))

    def get_patch(self, **kwargs):
        floor = PolygonPatch(self.shape, fc=kwargs.get('color', '#dddddd'),
                             ec=kwargs.get('color', '#111111'),
                             zorder=ZORDERS['floor'])
        carpet = self.carpet.get_patch(**kwargs)
        return [floor] + carpet

    def add_furniture(self, *furniture):
        for f in furniture:
            f.room = self
            self.furniture.append(f)

    def are_all_furniture_inside(self):
        res = []
        for f in self.furniture:
            if isinstance(f, Door):
                res.append(True)
            else:
                res.append(self.is_furniture_inside(f))
        return all(res)

    def is_furniture_inside(self, furniture):
        return self.shape.contains(furniture.occupied_area)

    def has_tv(self):
        return any([isinstance(f, Tv) for f in self.furniture])

    def get_tv(self):
        for f in self.furniture:
            if isinstance(f, Tv):
                return f
        return None

    def get_possible_carpet_radius(self):
        lowest_dist = np.inf
        for f in self.furniture:
            if not f.can_stand_on_carpet:
                lowest_dist = min(lowest_dist, self.carpet_center.distance(f.occupied_area))
        return lowest_dist

    def update_carpet_diameter(self):
        new_radius = self.get_possible_carpet_radius()
        self.carpet.diameter = new_radius * 2

    @property
    def params_to_optimize(self):
        params = []
        for f in self.furniture:
            if f.is_optimasible:
                params.append(f.params_to_optimize)
        return np.array(params)

    def apply_feature_vector(self, vector):
        matrix = vector.reshape(-1, 3)
        i = 0
        for f in self.furniture:
            if f.is_optimasible:
                f.set_params(matrix[i,0],matrix[i,1],matrix[i,2])
                f.update_polygon()
                i+=1

    def are_furniture_ok(self):
        are_ok = True
        for f1 in self.furniture:
            for f2 in self.furniture:
                if f1 == f2:
                    continue
                are_ok = (are_ok
                          and (not isinstance(f1, Window),
                               self.is_furniture_inside(f1))  # windows excluded due to exceptional hit box
                          and not f1.intersects(f2))
                if not are_ok:
                    return False
        return True


class RoomDrawer:
    def __init__(self, room, figsize=(16, 16)):
        plt.ion()
        self.room = room
        self.figure = plt.figure(1, figsize=figsize)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(xmin=0, xmax=self.room.width)
        self.ax.set_ylim(ymin=0, ymax=self.room.height)

    def draw_all(self, **kwargs):
        self.room.draw(self.ax)
        for furniture in self.room.furniture:
            furniture.draw(self.ax, **kwargs)
        self.figure.canvas.draw()
        self.figure.show()

    def update(self):
        self.figure.canvas.draw()
