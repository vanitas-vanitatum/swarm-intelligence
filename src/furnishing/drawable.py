from abc import ABC, abstractmethod


class Drawable(ABC):
    @abstractmethod
    def get_patch(self, **kwargs):
        pass

    def draw(self, ax, **kwargs):
        patch = self.get_patch(**kwargs)
        if isinstance(patch, list) or isinstance(patch, tuple):
            for p in patch:
                ax.add_patch(p)
        else:
            ax.add_patch(patch)
