from abc import ABC, abstractmethod

import numpy as np
import shapely
import shapely.affinity
import shapely.geometry
from descartes import PolygonPatch
from shapely.geometry import Polygon, Point

from src.common import ZORDERS, SHOW_HIT_BOXES, get_hit_box_visualization
from src.furnishing.drawable import Drawable
from src.furnishing.math_utils import MatrixUtils

FURNITURE_COLORS = {
    'skin': '#8d5524',
    'gray': '#aaaaaa'
}


class BaseFurniture(Drawable, ABC):
    def __init__(self, x, y, angle, can_stand_on_carpet, is_optimasible, name=None):
        self.params = np.array([x, y, angle])
        self.shape = None
        self.hit_box_shape = None
        self.room = None
        self.is_optimasible = is_optimasible
        self.base_color = np.array((0, 0, 0))
        self.can_stand_on_carpet = can_stand_on_carpet
        self.name = name or self.__class__.__name__.lower()

    def update_params(self, update):
        if self.is_optimasible:
            self.params += update
        else:
            raise ValueError('Furniture is not optimasible')

    def set_params(self, x, y, angle):
        self.params = np.array([x, y, angle])

    @property
    def x(self):
        return self.params[0]

    @property
    def y(self):
        return self.params[1]

    @property
    def angle(self):
        return self.params[2]

    @x.setter
    def x(self, x):
        self.params[0] = x

    @y.setter
    def y(self, y):
        self.params[1] = y

    @angle.setter
    def angle(self, angle):
        self.params[2] = angle

    @property
    def center(self):
        return self.params[0:2]

    @property
    def params_to_optimize(self):
        return self.params

    @abstractmethod
    def get_patch(self, **kwargs):
        pass

    @abstractmethod
    def update_polygon(self):
        pass

    def intersects(self, other):
        return self.hit_box_shape.intersects(other.hit_box_shape)

    def is_inside_room(self):
        if self.room is None:
            return
        return self.shape.within(self.room)

    @property
    def occupied_area(self):
        return self.hit_box_shape

    @property
    def color(self):
        # https://stackoverflow.com/questions/3380726/converting-a-rgb-color-tuple-to-a-six-digit-code-in-python
        return '#%02x%02x%02x' % tuple(self.base_color)

    def paint(self, new_color):
        assert len(new_color) == 3
        self.base_color = np.array(new_color)


class RectangularFurniture(BaseFurniture):

    def __init__(self, x, y, angle, width, height, can_stand_on_carpet, is_optimasible, name=None):
        self.points = np.array([(x - width / 2, y - height / 2),
                                (x + width / 2, y - height / 2),
                                (x + width / 2, y + height / 2),
                                (x - width / 2, y + height / 2)])
        super().__init__(x, y, angle, can_stand_on_carpet, is_optimasible, name)
        self.width = width
        self.height = height

        self.update_polygon()

    def get_patch(self, **kwargs):
        patch = [PolygonPatch(self.shape, fc=kwargs.get('color', FURNITURE_COLORS['skin']),
                              ec=kwargs.get('color', FURNITURE_COLORS['gray']),
                              zorder=ZORDERS['furniture'])]
        if SHOW_HIT_BOXES:
            patch.append(get_hit_box_visualization(self.hit_box_shape))
        return patch

    def update_polygon(self):
        self.points = np.array([(self.x - self.width / 2, self.y - self.height / 2),
                                (self.x + self.width / 2, self.y - self.height / 2),
                                (self.x + self.width / 2, self.y + self.height / 2),
                                (self.x - self.width / 2, self.y + self.height / 2)])
        new_points = MatrixUtils.rotate_points(self.points, self.angle)
        self.shape = Polygon(new_points)
        self.hit_box_shape = self.shape


class EllipseFurniture(BaseFurniture):
    def __init__(self, x, y, angle, width, height, can_stand_on_carpet, is_optimasible, name=None):
        super().__init__(x, y, angle, can_stand_on_carpet, is_optimasible, name)
        self.base_point = (x, y)
        self.width = width
        self.height = height

        self.update_polygon()

    def get_patch(self, **kwargs):
        patch = PolygonPatch(self.shape, fc=kwargs.get('color', FURNITURE_COLORS['skin']),
                             ec=kwargs.get('color', FURNITURE_COLORS['gray']))
        return patch

    def update_polygon(self):
        self.base_color = (self.x, self.y)
        circ = Point(self.base_point).buffer(1)
        circ = shapely.affinity.scale(circ, self.width / 2, self.height / 2)
        circ = shapely.affinity.rotate(circ, self.angle)
        self.shape = circ
        self.hit_box_shape = self.shape


class RoundedFurniture(EllipseFurniture):
    def __init__(self, x, y, diameter, can_stand_on_carpet, is_optimasible, name=None):
        super().__init__(x, y, 0, diameter, diameter, can_stand_on_carpet, is_optimasible, name)


class RoundedCornersFurniture(RectangularFurniture):

    def __init__(self, x, y, angle, width, height, can_stand_on_carpet, is_optimasible, name=None):
        super().__init__(x, y, angle, width, height, can_stand_on_carpet, is_optimasible, name)

    def update_polygon(self):
        super().update_polygon()
        self.shape = self.shape.buffer(-self.width / 3).buffer(self.width / 3)
        self.hit_box_shape = self.shape
