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
