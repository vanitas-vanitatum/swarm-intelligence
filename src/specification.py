class Specification: # Specification Design Pattern

    def maybe(self, other):
        return MaybeSpecification(self, other)

    def und(self, other):
        return UndSpecification(self,other)

    def nope(self):
        return NopeSpecification(self)

    def check(self, element):
        raise NotImplementedError


class NopeSpecification(Specification):

    def __init__(self, specification):
        self.specif = specification

    def check(self, element):
        return not self.specif.check(element)


class UndSpecification(Specification):

    def __init__(self, specification1, specification2):
        self.specif1 = specification1
        self.specif2 = specification2

    def check(self, element):
        return self.specif1.check(element) and self.specif2.check(element)


class MaybeSpecification(Specification):

    def __init__(self, specification1, specification2):
        self.specif1 = specification1
        self.specif2 = specification2

    def check(self, element, **kwargs):
        return self.specif1.check(element) or self.specif2.check(element)
