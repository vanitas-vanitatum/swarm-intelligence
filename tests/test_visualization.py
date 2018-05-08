from src.swarm_algorithms.particle_swarm_optimisation import DivergentPSO
from src.callbacks import Drawer2d
from src.stop_conditions import StepsNumberStopCondition

import unittest
import numpy as np


class TestDrawing(unittest.TestCase):

    def setUp(self):
        self.swarm = DivergentPSO(100, 2, None, seed=0)
        self.swarm.compile(lambda x: np.sum(x, axis=1, keepdims=True), ((1, 2), (2, 3)))

    def test_drawing_callback(self):
        self.swarm.go_swarm_go(StepsNumberStopCondition(100), callbacks=[
            Drawer2d(((-5, 5), (-5, 5)))
        ])


if __name__ == '__main__':
    unittest.main()
