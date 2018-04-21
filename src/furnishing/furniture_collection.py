import numpy as np
from descartes import PolygonPatch
from shapely.geometry import Polygon

from src.furnishing.furniture_construction import RectangularFurniture
from src.furnishing.math_utils import MatrixUtils


def get_angle_to_fit_wall(x, y, room_width, room_height):
    if y == 0:
        angle = 0
    elif y == room_height:
        angle = 180
    elif x == 0:
        angle = 90
    elif x == room_width:
        angle = 270
    else:
        raise ValueError('Inappropriate positioning')
    return angle


class Window(RectangularFurniture):
    def __init__(self, x, y, room_width, room_height, width, view_angle=30, view_depth=3):
        self.hit_box_points = None
        self.view_angle = view_angle
        self.view_depth = view_depth

        super().__init__(x, y, get_angle_to_fit_wall(x, y, room_width, room_height),
                         width, 0.5, False)

    def get_patch(self, **kwargs):
        patch = PolygonPatch(self.shape, fc=kwargs.get('window_color', 'black'),
                             ec=kwargs.get('window_color', 'black'), alpha=0.3)
        return patch

    def update_polygon(self):
        """     _______  <- upper y
               /       \
              /         \
             /___________\ <- lower y
            <------------->
              lower_width
        :return: None
        """
        lower_width = 2 * self.view_depth * np.tan(np.deg2rad(self.view_angle)) + self.width

        if self.angle == 0:
            lower_left_point = (self.x - self.width / 2, self.y)
            upper_left_point = (self.x - lower_width / 2, self.y + self.view_depth)
            upper_right_point = (self.x + lower_width / 2, self.y + self.view_depth)
            lower_right_point = (self.x + self.width / 2, self.y)

        elif self.angle == 90:
            lower_left_point = (self.x, self.y - self.width / 2)
            upper_left_point = (self.x, self.y + self.width / 2)
            upper_right_point = (self.x + self.view_depth, self.y + lower_width / 2)
            lower_right_point = (self.x + self.view_depth, self.y - lower_width / 2)

        elif self.angle == 180:
            lower_left_point = (self.x - self.width / 2, self.y)
            upper_left_point = (self.x - lower_width / 2, self.y - self.view_depth)
            upper_right_point = (self.x + lower_width / 2, self.y - self.view_depth)
            lower_right_point = (self.x + self.width / 2, self.y)

        elif self.angle == 270:
            lower_left_point = (self.x, self.y - self.width / 2)
            upper_left_point = (self.x, self.y + self.width / 2)
            upper_right_point = (self.x - self.view_depth, self.y + lower_width / 2)
            lower_right_point = (self.x - self.view_depth, self.y - lower_width / 2)

        else:
            raise ValueError('Incorrect angle')

        self.hit_box_points = np.array((
            lower_left_point,
            upper_left_point,
            upper_right_point,
            lower_right_point,
        ))
        new_points = MatrixUtils.rotate_points(self.points, self.angle)
        self.shape = Polygon(new_points)
        self.hit_box_shape = Polygon(self.hit_box_points)


class Door(RectangularFurniture):
    def __init__(self, x, y, room_width, room_height, width):
        super().__init__(x, y, get_angle_to_fit_wall(x, y, room_width, room_height),
                         width, 0.5, False)

    def get_patch(self, **kwargs):
        patch = PolygonPatch(self.shape, fc=kwargs.get('door_color', 'black'),
                             ec=kwargs.get('door_color', 'black'), alpha=0.9)
        return patch

    def update_polygon(self):
        new_points = MatrixUtils.rotate_points(self.points, self.angle)
        self.shape = Polygon(new_points)
        self.hit_box_shape = self.shape


class Carpet(RectangularFurniture):
    pass
