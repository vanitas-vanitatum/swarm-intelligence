import numpy as np
from descartes import PolygonPatch
from shapely.geometry import Polygon, Point

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
    def __init__(self, x, y, room_width, room_height, width, open_angle=120):
        assert 0 < open_angle < 180, "Rly? Do you live in Wallmart?"

        self.open_angle = open_angle
        self.hit_box_points_door_1 = np.array([
            [x - width / 2, y],
            [x - width / 2, y + width],
            [x + width * 1.5, y + width],
            [x + width * 1.5, y]
        ])
        self.hit_box_points_door_2 = np.array([
            [x - width / 2, y - width],
            [x - width / 2, y],
            [x + width * 1.5, y],
            [x + width * 1.5, y]
        ])

        self.pivot_hit_box_points = np.array([
            [x + width / 2, y]
        ])

        self.hit_box_points_door_1 = MatrixUtils.rotate_points(
            self.hit_box_points_door_1, self.open_angle, self.pivot_hit_box_points
        )

        super().__init__(x, y, get_angle_to_fit_wall(x, y, room_width, room_height),
                         width, 0.5, False)

    def get_patch(self, **kwargs):
        patch = PolygonPatch(self.shape, fc=kwargs.get('door_color', 'black'),
                             ec=kwargs.get('door_color', 'black'), alpha=0.9)
        return patch

    def update_polygon(self):
        self.pivot_hit_box_points = MatrixUtils.rotate_points(self.pivot_hit_box_points,
                                                              self.angle,
                                                              np.array([self.x, self.y]))
        self.hit_box_points_door_1 = MatrixUtils.rotate_points(self.hit_box_points_door_1,
                                                               self.angle,
                                                               np.array([self.x, self.y]))
        self.hit_box_points_door_2 = MatrixUtils.rotate_points(self.hit_box_points_door_2,
                                                               self.angle,
                                                               np.array([self.x, self.y]))

        polygon_hit_box_points_door_1 = Polygon(self.hit_box_points_door_1)
        polygon_hit_box_points_door_2 = Polygon(self.hit_box_points_door_2)

        self.hit_box_shape = Point(self.pivot_hit_box_points[0]).buffer(self.width)
        self.hit_box_shape = self.hit_box_shape.difference(polygon_hit_box_points_door_1).difference(polygon_hit_box_points_door_2)
        new_points = MatrixUtils.rotate_points(self.points, self.angle)
        self.shape = Polygon(new_points)
        self.hit_box_shape = self.shape


class Carpet(RectangularFurniture):
    pass
