import numpy as np
import math
from src.specification import Specification


class CustomConstraint(Specification):

    def __init__(self, check_function):
        self.check_function = check_function

    def check(self, element):
        return self.check_function(element)


class LessThanConstraint(Specification):

    def __init__(self, limit):
        self.limit = limit

    def check(self, element):
        return element < self.limit

