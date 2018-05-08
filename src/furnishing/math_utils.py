import numpy as np


class MatrixUtils:
    @staticmethod
    def rotate_points(points, angle, rotate_point=None):
        """
        Creates rotate matrix and rotates points. Note that most of the time you will need points in base form
        (not rotated).
        :param points: np.ndarray of points [(x1, y1), ..., (xn, yn)].
        :param angle: Angle in degrees to use in the rotation matrix.
        :param rotate_point: Point which is used as pivot to rotate figure.
        :return: Rotated points as np.ndarray.
        """
        if rotate_point is None:
            rotate_point = np.mean(points, axis=0, keepdims=True)

        angle = np.deg2rad(angle)

        rotation_matrix = [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ]

        rotation_matrix = np.array(rotation_matrix)
        normalized = points - rotate_point
        new_rotation = normalized.dot(rotation_matrix)
        restored = new_rotation + rotate_point
        return restored

    @staticmethod
    def get_angle_between_vectors(vector_1, vector_2):
        """
        Standard math equation.
        :param vector_1: Vector in form [x, y].
        :param vector_2: Vector in form [x, y].
        :return: Angle in degrees between vectors.
        """
        length_1 = np.linalg.norm(vector_1, ord=2)
        length_2 = np.linalg.norm(vector_2, ord=2)
        cos_angle = vector_1.dot(vector_2.T) / (length_1 * length_2)
        radian_angle = np.arccos(cos_angle)
        return np.rad2deg(radian_angle)
