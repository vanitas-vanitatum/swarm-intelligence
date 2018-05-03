import warnings

import numpy as np
from descartes import PolygonPatch
from shapely.geometry import Polygon, Point

from src.common import SHOW_HIT_BOXES, ZERO_POINT, ZORDERS, get_hit_box_visualization
from src.furnishing.furniture_construction import RectangularFurniture, RoundedFurniture, FURNITURE_COLORS
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
    def __init__(self, x, y, room_width, room_height, width, view_angle=30, view_depth=3, name=None):
        self.hit_box_points = None
        self.view_angle = view_angle
        self.view_depth = view_depth

        super().__init__(x, y, get_angle_to_fit_wall(x, y, room_width, room_height),
                         width, 0.5, True, False, name)

    def get_patch(self, **kwargs):
        patch = [PolygonPatch(self.shape,
                              fc=kwargs.get('window_color', 'blue'),
                              ec=kwargs.get('window_color', 'blue'), alpha=1)]
        if SHOW_HIT_BOXES:
            patch.append(get_hit_box_visualization(self.hit_box_shape))

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
    def __init__(self, x, y, room_width, room_height, width, open_angle=120, name=None):
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
                         width, 0.5, True, False, name)

    def get_patch(self, **kwargs):
        patch = [PolygonPatch(self.shape,
                              fc=kwargs.get('door_color', 'black'),
                              ec=kwargs.get('door_color', 'black'), alpha=1)]
        if SHOW_HIT_BOXES:
            patch.append(get_hit_box_visualization(self.hit_box_shape))
        return patch

    def update_polygon(self):
        self.pivot_hit_box_points = MatrixUtils.rotate_points(self.pivot_hit_box_points,
                                                              self.angle,
                                                              self.center)
        self.hit_box_points_door_1 = MatrixUtils.rotate_points(self.hit_box_points_door_1,
                                                               self.angle,
                                                               self.center)
        self.hit_box_points_door_2 = MatrixUtils.rotate_points(self.hit_box_points_door_2,
                                                               self.angle,
                                                               self.center)

        polygon_hit_box_points_door_1 = Polygon(self.hit_box_points_door_1)
        polygon_hit_box_points_door_2 = Polygon(self.hit_box_points_door_2)

        self.hit_box_shape = Point(self.pivot_hit_box_points[0]).buffer(self.width)
        self.hit_box_shape = self.hit_box_shape.difference(polygon_hit_box_points_door_1).difference(
            polygon_hit_box_points_door_2)
        new_points = MatrixUtils.rotate_points(self.points, self.angle)
        self.shape = Polygon(new_points)


class Cupboard(RectangularFurniture):
    def __init__(self, x, y, angle, width, height, can_stand_on_carpet, open_angle=120, name=None):
        assert 0 < open_angle < 270, "Can't open doors through cupboard wall"

        self.open_angle = open_angle

        self.pivot_door_1 = np.array([
            [x + width / 2, y + height / 2]
        ])
        self.pivot_door_2 = np.array([
            [x - width / 2, y + height / 2]
        ])

        self.difference_rectangles_door_1 = [
            np.array([
                [self.pivot_door_1[0, 0] - width / 2, self.pivot_door_1[0, 1]],
                [self.pivot_door_1[0, 0] - width / 2, self.pivot_door_1[0, 1] - width / 2],
                [self.pivot_door_1[0, 0] + width / 2, self.pivot_door_1[0, 1] - width / 2],
                [self.pivot_door_1[0, 0] + width / 2, self.pivot_door_1[0, 1]]
            ]),
            np.array([
                [self.pivot_door_1[0, 0] - width / 2, self.pivot_door_1[0, 1]],
                [self.pivot_door_1[0, 0] - width / 2, self.pivot_door_1[0, 1] + width / 2],
                [self.pivot_door_1[0, 0] + width / 2, self.pivot_door_1[0, 1] + width / 2],
                [self.pivot_door_1[0, 0] + width / 2, self.pivot_door_1[0, 1]]
            ])
        ]

        self.difference_rectangles_door_2 = [
            np.array([
                [self.pivot_door_2[0, 0] - width / 2, self.pivot_door_2[0, 1]],
                [self.pivot_door_2[0, 0] - width / 2, self.pivot_door_2[0, 1] - width / 2],
                [self.pivot_door_2[0, 0] + width / 2, self.pivot_door_2[0, 1] - width / 2],
                [self.pivot_door_2[0, 0] + width / 2, self.pivot_door_2[0, 1]]
            ]),
            np.array([
                [self.pivot_door_2[0, 0] - width / 2, self.pivot_door_2[0, 1]],
                [self.pivot_door_2[0, 0] - width / 2, self.pivot_door_2[0, 1] + width / 2],
                [self.pivot_door_2[0, 0] + width / 2, self.pivot_door_2[0, 1] + width / 2],
                [self.pivot_door_2[0, 0] + width / 2, self.pivot_door_2[0, 1]]
            ])
        ]

        self.difference_rectangles_door_1[1] = MatrixUtils.rotate_points(
            self.difference_rectangles_door_1[1], self.open_angle, self.pivot_door_1
        )

        self.difference_rectangles_door_2[1] = MatrixUtils.rotate_points(
            self.difference_rectangles_door_2[1], -self.open_angle, self.pivot_door_2
        )

        super().__init__(x, y, angle, width, height, can_stand_on_carpet, True, name)

    def get_patch(self, **kwargs):
        patch = [PolygonPatch(self.shape,
                              fc=kwargs.get('cupboard_color', FURNITURE_COLORS['skin']),
                              ec=kwargs.get('cupboard_color', FURNITURE_COLORS['gray']), alpha=1,
                              zorder=ZORDERS['furniture'])]
        if SHOW_HIT_BOXES:
            patch.append(get_hit_box_visualization(self.hit_box_shape))
        return patch

    def update_polygon(self):
        self.pivot_door_1 = MatrixUtils.rotate_points(self.pivot_door_1,
                                                      self.angle,
                                                      self.center)
        self.pivot_door_2 = MatrixUtils.rotate_points(self.pivot_door_2,
                                                      self.angle,
                                                      self.center)

        door_1_shape = Point(self.pivot_door_1[0]).buffer(self.width / 2)
        door_2_shape = Point(self.pivot_door_2[0]).buffer(self.width / 2)

        for i in range(len(self.difference_rectangles_door_1)):
            self.difference_rectangles_door_1[i] = MatrixUtils.rotate_points(
                self.difference_rectangles_door_1[i],
                self.angle,
                self.center
            )

            self.difference_rectangles_door_2[i] = MatrixUtils.rotate_points(
                self.difference_rectangles_door_2[i],
                self.angle,
                self.center
            )

            polygon_hit_box_points_door_1 = Polygon(self.difference_rectangles_door_1[i])
            polygon_hit_box_points_door_2 = Polygon(self.difference_rectangles_door_2[i])

            door_1_shape = door_1_shape.difference(polygon_hit_box_points_door_1)
            door_2_shape = door_2_shape.difference(polygon_hit_box_points_door_2)

        new_points = MatrixUtils.rotate_points(self.points, self.angle)
        self.shape = Polygon(new_points)
        self.hit_box_shape = door_1_shape.union(door_2_shape).union(self.shape)


class Sofa(RectangularFurniture):
    def __init__(self, x, y, angle, width, height, can_stand_on_carpet, allowed_tv_angle=30, name=None):
        self.sight_vector = np.array(
            [0, height], dtype=np.float32
        )

        self.sight_vector /= np.linalg.norm(self.sight_vector, ord=2)
        self.allowed_tv_angle = allowed_tv_angle
        super().__init__(x, y, angle, width, height, can_stand_on_carpet, True, name)

    def update_polygon(self):
        super().update_polygon()
        self.sight_vector = MatrixUtils.rotate_points(
            self.sight_vector, self.angle, ZERO_POINT
        )

    def can_see_tv(self, tv):
        if self.room is None:
            warnings.warn("Sofa is not in the room")
            return
        if tv.room is None:
            warnings.warn("No TV found in the room")
            return
        return (self._is_normal_face_visible(tv.tv_normal_vector, tv.center) and
                self._are_edge_points_visible(tv.get_edge_tv_points()))

    def _is_normal_face_visible(self, tv_normal_vector, tv_center):
        tv_to_sofa_vector = self.center - tv_center
        angle = MatrixUtils.get_angle_between_vectors(tv_to_sofa_vector, tv_normal_vector)

        return angle <= self.allowed_tv_angle

    def _are_edge_points_visible(self, tv_edge_points):
        def _check_for_one_point(point, sofa_center, sofa_normal_vector, allowed_angle):
            center_to_edge_point_vector = point - sofa_center
            angle = MatrixUtils.get_angle_between_vectors(sofa_normal_vector, center_to_edge_point_vector)
            return angle <= allowed_angle

        return (_check_for_one_point(tv_edge_points[0], self.center, self.sight_vector, self.allowed_tv_angle) and
                _check_for_one_point(tv_edge_points[1], self.center, self.sight_vector, self.allowed_tv_angle))


class Tv(RectangularFurniture):
    def __init__(self, x, y, angle, width, height, tv_width, tv_thickness, can_stand_on_carpet, name=None):
        assert tv_width <= width
        self.tv_width = tv_width
        self.tv_thickness = tv_thickness
        self.tv_points = np.array([
            [x - tv_width / 2, y + tv_thickness / 2],
            [x - tv_width / 2, y - tv_thickness / 2],
            [x + tv_width / 2, y - tv_thickness / 2],
            [x + tv_width / 2, y + tv_thickness / 2]
        ])
        self.tv_shape = Polygon(self.tv_points)
        self.tv_normal_vector = np.array([
            [0, tv_thickness]
        ], dtype=np.float32)
        self.tv_normal_vector /= np.linalg.norm(self.tv_normal_vector, ord=2)

        super().__init__(x, y, angle, width, height, can_stand_on_carpet, True, name)

    def get_patch(self, **kwargs):
        patch = PolygonPatch(self.shape, fc=kwargs.get(self.name, FURNITURE_COLORS['skin']),
                             ec=kwargs.get(self.name, FURNITURE_COLORS['gray']), alpha=1,
                             zorder=ZORDERS['furniture'])
        tv_patch = PolygonPatch(self.tv_shape, fc=kwargs.get(self.name + '_tv', 'black'),
                                ec=kwargs.get(self.name + '_tv', 'black'),
                                alpha=0.5, zorder=ZORDERS['things on furniture'])

        patches = [patch, tv_patch]
        if SHOW_HIT_BOXES:
            patches.append(get_hit_box_visualization(self.hit_box_shape))
        return patches

    def update_polygon(self):
        super().update_polygon()
        self.tv_points = np.array([(self.x - self.tv_width / 2, self.y + self.tv_thickness / 2),
                                   (self.x - self.tv_width / 2, self.y - self.tv_thickness / 2),
                                   (self.x + self.tv_width / 2, self.y - self.tv_thickness / 2),
                                   (self.x + self.tv_width / 2, self.y + self.tv_thickness / 2)])
        new_points = MatrixUtils.rotate_points(self.tv_points, self.angle)
        self.tv_points = new_points
        self.tv_shape = Polygon(new_points)
        self.tv_normal_vector = MatrixUtils.rotate_points(self.tv_normal_vector, self.angle, ZERO_POINT)

    def get_edge_tv_points(self):
        return self.tv_points[[0, 3]]


class Carpet(RoundedFurniture):
    def __init__(self, room_width, room_height, diameter):
        super().__init__(room_width / 2, room_height / 2, diameter, False, False, "carpet")

    def get_patch(self, **kwargs):
        patch = super().get_patch(**kwargs)
        patch.zorder = ZORDERS['carpet']
        patch.set_facecolor(kwargs.get('carpet_color', 'white'))
        patch.set_edgecolor(kwargs.get('carpet_edge_color', 'gray'))
        patch.set_alpha(1)
        return patch

    @property
    def area(self):
        return self.shape.area

    @property
    def diameter(self):
        return self.width

    @diameter.setter
    def diameter(self, new_diameter):
        self.width = new_diameter
        self.height = new_diameter
        self.update_polygon()
