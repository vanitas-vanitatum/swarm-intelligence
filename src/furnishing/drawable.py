from abc import ABC, abstractmethod


class Drawable(ABC):
    @abstractmethod
    def get_patch(self, **kwargs):
        pass

    def draw(self, ax, **kwargs):
        patch = self.get_patch(**kwargs)
        ax.add_patch(patch)
