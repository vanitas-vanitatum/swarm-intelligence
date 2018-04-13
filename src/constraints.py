import numpy as np
import math
from src.specification import Specification


class CustomConstraint(Specification):

    def __init__(self, check_function):
        self.check_function = check_function

    def check(self, solution):
        return self.check_function(solution)


class LessThanConstraint(Specification):

    def __init__(self, limit):
        self.limit = limit

    def check(self, solution):
        return all(solution < self.limit)

