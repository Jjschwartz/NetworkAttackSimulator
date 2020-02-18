import unittest
import numpy as np
from nasim.env.environment import NASimEnv


class StateTestCase(unittest.TestCase):

    def setUp(self):
        self.E = 1
        self.M = 3
        self.env = self.get_env(self.M, self.E)
        self.init_state = self.env.reset()

    """ Helper method """
    def get_env(self, nM, nS):
        env = NASimEnv.from_params(nM, nS)
        return env

    def test_state_copy(self):
        state_copy = self.init_state.copy()
        self.assertFalse(state_copy is self.init_state)
        self.assertTrue(state_copy == self.init_state)

    def test_state_flatten(self):
        expected_flat = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0]])
        actual_flat = self.init_state.flatten()
        self.assertTrue(np.equal(expected_flat, actual_flat).all())


if __name__ == "__main__":
    unittest.main()
