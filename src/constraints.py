import numpy as np
import math
from src.specification import Specification


class BaseConstraint:

    def maybe(self, other):
        return MaybeConstraint(self, other)

    def und(self, other):
        return UndConstraint(self,other)

    def nope(self):
        return NopeConstraint(self)

    def check(self, element):
        raise NotImplementedError


class NopeConstraint(BaseConstraint):

    def __init__(self, constraint):
        self.constraint = constraint

    def check(self, element):
        return ~self.constraint.check(element)


class UndConstraint(BaseConstraint):

    def __init__(self, constraint1, constraint2):
        self.constraint1 = constraint1
        self.constraint2 = constraint2

    def check(self, element):
        return self.constraint1.check(element) & self.constraint2.check(element)


class MaybeConstraint(BaseConstraint):

    def __init__(self, constraint1, constraint2):
        self.constraint1 = constraint1
        self.constraint2 = constraint2

    def check(self, element, **kwargs):
        return self.constraint1.check(element) | self.constraint2.check(element)


class NoneConstraint(BaseConstraint):

    def check(self, solution):
        return True


class CustomConstraint(BaseConstraint):

    def __init__(self, check_function):
        self.check_function = check_function

    def check(self, solution):
        return self.check_function(solution)


class LessThanConstraint(BaseConstraint):

    def __init__(self, limit):
        self.limit = limit

    def check(self, solution):
        #print(solution.shape)
        return np.all(solution < self.limit, axis=1)


class MoreThanConstraint(BaseConstraint):

    def __init__(self, limit):
        self.limit = limit

    def check(self, solution):
        #print(solution.shape)
        return np.all(solution > self.limit, axis=1)

