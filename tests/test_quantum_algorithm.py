import unittest
import numpy as np

from src.swarm_algorithms.quantum_algorithm import QuantumDeltaParticleSwarmOptimization


class TestQuantumAlgorithm(unittest.TestCase):
    def setUp(self):
        self.alg = QuantumDeltaParticleSwarmOptimization(100, 2, None, np.log(np.sqrt(2)), seed=0)
        self.alg.compile(lambda x: np.sum(x, axis=1, keepdims=True), ((1, 2), (2, 3)))

    def test_init(self):
        self.assertEqual((self.alg.population_size, self.alg.nb_features), self.alg.population.shape)
        self.assertTrue(self.alg._compiled)

        self.assertTrue(all(self.alg.population[:, 0] <= 2))
        self.assertTrue(all(self.alg.population[:, 0] >= 1))
        self.assertTrue(all(self.alg.population[:, 1] <= 3))
        self.assertTrue(all(self.alg.population[:, 1] >= 2))

    def test_get_new_positions(self):
        for i in range(100):
            position = self.alg.get_new_positions(i)
            self.assertEqual((self.alg.population_size, self.alg.nb_features), position.shape)

    def test_update_position(self):
        for i in range(10):
            position_movement = self.alg.get_new_positions(i)
            self.alg.update_positions(position_movement, i)
            np.testing.assert_almost_equal(self.alg.population, position_movement, decimal=5)


if __name__ == '__main__':
    unittest.main()
