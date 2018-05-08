import numpy as np
from descartes import PolygonPatch

SHOW_HIT_BOXES = True
ZERO_POINT = np.array([0, 0])

ZORDERS = {
    'floor': -99,
    'carpet': -89,
    'furniture': 0,
    'things on furniture': 1,
    'hit box': 99
}

HIT_BOX_COLOR = 'green'


def get_hit_box_visualization(hit_box):
    patch = PolygonPatch(hit_box,
                         fc=HIT_BOX_COLOR,
                         ec=HIT_BOX_COLOR, alpha=0.5,
                         zorder=ZORDERS['hit box'])
    return patch
